//===- FunctionFiltering.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements transforms to filter out functions intended for the host
// when compiling for the device and vice versa.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace flangomp {
#define GEN_PASS_DEF_FUNCTIONFILTERINGPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

using namespace mlir;

namespace {
class FunctionFilteringPass
    : public flangomp::impl::FunctionFilteringPassBase<FunctionFilteringPass> {
public:
  FunctionFilteringPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OpBuilder opBuilder(context);
    auto op = dyn_cast<omp::OffloadModuleInterface>(getOperation());
    if (!op || !op.getIsTargetDevice())
      return;

    op->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
      // Do not filter functions with target regions inside, because they have
      // to be available for both host and device so that regular and reverse
      // offloading can be supported.
      bool hasTargetRegion =
          funcOp
              ->walk<WalkOrder::PreOrder>(
                  [&](omp::TargetOp) { return WalkResult::interrupt(); })
              .wasInterrupted();

      omp::DeclareTargetDeviceType declareType =
          omp::DeclareTargetDeviceType::host;
      auto declareTargetOp =
          dyn_cast<omp::DeclareTargetInterface>(funcOp.getOperation());
      if (declareTargetOp && declareTargetOp.isDeclareTarget())
        declareType = declareTargetOp.getDeclareTargetDeviceType();

      // Filtering a function here means deleting it if it doesn't contain a
      // target region. Else we explicitly set the omp.declare_target
      // attribute. The second stage of function filtering at the MLIR to LLVM
      // IR translation level will remove functions that contain the target
      // region from the generated llvm IR.
      if (declareType == omp::DeclareTargetDeviceType::host) {
        SymbolTable::UseRange funcUses = *funcOp.getSymbolUses(op);
        for (SymbolTable::SymbolUse use : funcUses) {
          Operation *callOp = use.getUser();
          if (auto internalFunc = mlir::dyn_cast<func::FuncOp>(callOp)) {
            // Do not delete internal procedures holding the symbol of their
            // Fortran host procedure as attribute.
            internalFunc->removeAttr(fir::getHostSymbolAttrName());
            // Set public visibility so that the function is not deleted by MLIR
            // because unused. Changing it is OK here because the function will
            // be deleted anyway in the second filtering phase.
            internalFunc.setVisibility(mlir::SymbolTable::Visibility::Public);
            continue;
          }
          // If the callOp has users then replace them with Undef values.
          if (!callOp->use_empty()) {
            SmallVector<Value> undefResults;
            for (Value res : callOp->getResults()) {
              opBuilder.setInsertionPoint(callOp);
              undefResults.emplace_back(
                  opBuilder.create<fir::UndefOp>(res.getLoc(), res.getType()));
            }
            callOp->replaceAllUsesWith(undefResults);
          }
          // Remove the callOp
          callOp->erase();
        }
        if (!hasTargetRegion) {
          funcOp.erase();
          return WalkResult::skip();
        }

        if (failed(rewriteHostRegion(funcOp.getRegion()))) {
          funcOp.emitOpError() << "could not be rewritten for target device";
          return WalkResult::interrupt();
        }

        if (declareTargetOp)
          declareTargetOp.setDeclareTarget(declareType,
                                           omp::DeclareTargetCaptureClause::to);
      }
      return WalkResult::advance();
    });
  }

private:
  /// Add the given \c omp.map.info to a sorted set while taking into account
  /// its dependencies.
  static void collectMapInfos(omp::MapInfoOp mapOp, Region &region,
                              llvm::SetVector<omp::MapInfoOp> &mapInfos) {
    for (Value member : mapOp.getMembers())
      collectMapInfos(cast<omp::MapInfoOp>(member.getDefiningOp()), region,
                      mapInfos);

    if (region.isAncestor(mapOp->getParentRegion()))
      mapInfos.insert(mapOp);
  }

  /// Add the given value to a sorted set if it should be replaced by a
  /// placeholder when used as a pointer-like argument to an operation
  /// participating in the initialization of an \c omp.map.info.
  static void markPtrOperandForRewrite(Value value,
                                       llvm::SetVector<Value> &rewriteValues) {
    // We don't need to rewrite operands if they are defined by block arguments
    // of operations that will still remain after the region is rewritten.
    if (isa<BlockArgument>(value) &&
        isa<func::FuncOp, omp::TargetDataOp>(
            cast<BlockArgument>(value).getOwner()->getParentOp()))
      return;

    rewriteValues.insert(value);
  }

  /// Rewrite the given host device region belonging to a function that contains
  /// \c omp.target operations, to remove host-only operations that are not used
  /// by device codegen.
  ///
  /// It is based on the expected form of the MLIR module as produced by Flang
  /// lowering and it performs the following mutations:
  ///   - Replace all values returned by the function with \c fir.undefined.
  ///   - Operations taking map-like clauses (e.g. \c omp.target,
  ///     \c omp.target_data, etc) are moved to the end of the function. If they
  ///     are nested inside of any other operations, they are hoisted out of
  ///     them. If the region belongs to \c omp.target_data, these operations
  ///     are hoisted to its top level, rather than to the parent function.
  ///   - Only \c omp.map.info operations associated to these target regions are
  ///     preserved. These are moved above all \c omp.target and sorted to
  ///     satisfy dependencies among them.
  ///   - \c bounds arguments are removed from \c omp.map.info operations.
  ///   - \c var_ptr and \c var_ptr_ptr arguments of \c omp.map.info are
  ///     handled as follows:
  ///     - \c var_ptr_ptr is expected to be defined by a \c fir.box_offset
  ///       operation which is preserved. Otherwise, the pass will fail.
  ///     - \c var_ptr can be defined by an \c hlfir.declare which is also
  ///       preserved. If the \c var_ptr or \c hlfir.declare \c memref argument
  ///       is a \c fir.address_of operation, that operation is also maintained.
  ///       Otherwise, it is replaced by a placeholder \c fir.alloca and a
  ///       \c fir.convert or kept unmodified when it is defined by an entry
  ///       block argument. If it has \c shape or \c typeparams arguments, they
  ///       are also replaced by applicable constants. \c dummy_scope arguments
  ///       are discarded.
  ///   - Every other operation not located inside of an \c omp.target is
  ///     removed.
  LogicalResult rewriteHostRegion(Region &region) {
    // Extract parent op information.
    auto [funcOp, targetDataOp] = [&region]() {
      Operation *parent = region.getParentOp();
      return std::make_tuple(dyn_cast<func::FuncOp>(parent),
                             dyn_cast<omp::TargetDataOp>(parent));
    }();
    assert((bool)funcOp != (bool)targetDataOp &&
           "region must be defined by either func.func or omp.target_data");

    // Collect operations that have mapping information associated to them.
    llvm::SmallVector<
        std::variant<omp::TargetOp, omp::TargetDataOp, omp::TargetEnterDataOp,
                     omp::TargetExitDataOp, omp::TargetUpdateOp>>
        targetOps;

    WalkResult result = region.walk<WalkOrder::PreOrder>([&](Operation *op) {
      // Skip the inside of omp.target regions, since these contain device code.
      if (auto targetOp = dyn_cast<omp::TargetOp>(op)) {
        targetOps.push_back(targetOp);
        return WalkResult::skip();
      }

      if (auto targetOp = dyn_cast<omp::TargetDataOp>(op)) {
        // Recursively rewrite omp.target_data regions as well.
        if (failed(rewriteHostRegion(targetOp.getRegion()))) {
          targetOp.emitOpError() << "rewrite for target device failed";
          return WalkResult::interrupt();
        }

        targetOps.push_back(targetOp);
        return WalkResult::skip();
      }

      if (auto targetOp = dyn_cast<omp::TargetEnterDataOp>(op))
        targetOps.push_back(targetOp);
      if (auto targetOp = dyn_cast<omp::TargetExitDataOp>(op))
        targetOps.push_back(targetOp);
      if (auto targetOp = dyn_cast<omp::TargetUpdateOp>(op))
        targetOps.push_back(targetOp);

      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      return failure();

    // Make a temporary clone of the parent operation with an empty region,
    // and update all references to entry block arguments to those of the new
    // region. Users will later either be moved to the new region or deleted
    // when the original region is replaced by the new.
    OpBuilder builder(&getContext());
    builder.setInsertionPointAfter(region.getParentOp());
    Operation *newOp = builder.cloneWithoutRegions(*region.getParentOp());
    Block &block = newOp->getRegion(0).emplaceBlock();

    llvm::SmallVector<Location> locs;
    locs.reserve(region.getNumArguments());
    llvm::transform(region.getArguments(), std::back_inserter(locs),
                    [](const BlockArgument &arg) { return arg.getLoc(); });
    block.addArguments(region.getArgumentTypes(), locs);

    for (auto [oldArg, newArg] :
         llvm::zip_equal(region.getArguments(), block.getArguments()))
      oldArg.replaceAllUsesWith(newArg);

    // Collect omp.map.info ops while satisfying interdependencies. This must be
    // updated whenever new map-like clauses are introduced or they are attached
    // to other operations.
    llvm::SetVector<omp::MapInfoOp> mapInfos;
    for (auto targetOp : targetOps) {
      std::visit(
          [&region, &mapInfos](auto op) {
            for (Value mapVar : op.getMapVars())
              collectMapInfos(cast<omp::MapInfoOp>(mapVar.getDefiningOp()),
                              region, mapInfos);

            if constexpr (std::is_same_v<decltype(op), omp::TargetOp>) {
              for (Value mapVar : op.getHasDeviceAddrVars())
                collectMapInfos(cast<omp::MapInfoOp>(mapVar.getDefiningOp()),
                                region, mapInfos);
            } else if constexpr (std::is_same_v<decltype(op),
                                                omp::TargetDataOp>) {
              for (Value mapVar : op.getUseDeviceAddrVars())
                collectMapInfos(cast<omp::MapInfoOp>(mapVar.getDefiningOp()),
                                region, mapInfos);
              for (Value mapVar : op.getUseDevicePtrVars())
                collectMapInfos(cast<omp::MapInfoOp>(mapVar.getDefiningOp()),
                                region, mapInfos);
            }
          },
          targetOp);
    }

    // Move omp.map.info ops to the new block and collect dependencies.
    llvm::SetVector<hlfir::DeclareOp> declareOps;
    llvm::SetVector<fir::BoxOffsetOp> boxOffsets;
    llvm::SetVector<Value> rewriteValues;
    for (omp::MapInfoOp mapOp : mapInfos) {
      // Handle var_ptr: hlfir.declare.
      if (auto declareOp = dyn_cast_if_present<hlfir::DeclareOp>(
              mapOp.getVarPtr().getDefiningOp())) {
        if (region.isAncestor(declareOp->getParentRegion()))
          declareOps.insert(declareOp);
      } else {
        markPtrOperandForRewrite(mapOp.getVarPtr(), rewriteValues);
      }

      // Handle var_ptr_ptr: fir.box_offset.
      if (Value varPtrPtr = mapOp.getVarPtrPtr()) {
        if (auto boxOffset = llvm::dyn_cast_if_present<fir::BoxOffsetOp>(
                varPtrPtr.getDefiningOp())) {
          if (region.isAncestor(boxOffset->getParentRegion()))
            boxOffsets.insert(boxOffset);
        } else {
          return mapOp->emitOpError() << "var_ptr_ptr rewrite only supported "
                                         "if defined by fir.box_offset";
        }
      }

      // Bounds are not used during target device codegen.
      mapOp.getBoundsMutable().clear();
      mapOp->moveBefore(&block, block.end());
    }

    // Create a temporary marker to simplify the op moving process below.
    builder.setInsertionPointToStart(&block);
    auto marker = builder.create<fir::UndefOp>(builder.getUnknownLoc(),
                                               builder.getNoneType());
    builder.setInsertionPoint(marker);

    // Move dependencies of hlfir.declare ops.
    for (hlfir::DeclareOp declareOp : declareOps) {
      Value memref = declareOp.getMemref();

      // If it's defined by fir.address_of, then we need to keep that op as well
      // because it might be pointing to a 'declare target' global.
      if (auto addressOf =
              dyn_cast_if_present<fir::AddrOfOp>(memref.getDefiningOp()))
        addressOf->moveBefore(marker);
      else
        markPtrOperandForRewrite(memref, rewriteValues);

      // Shape and typeparams aren't needed for target device codegen, but
      // removing them would break verifiers.
      Value zero;
      if (declareOp.getShape() || !declareOp.getTypeparams().empty())
        zero = builder.create<arith::ConstantOp>(declareOp.getLoc(),
                                                 builder.getI64IntegerAttr(0));

      if (auto shape = declareOp.getShape()) {
        Operation *shapeOp = shape.getDefiningOp();
        unsigned numArgs = shapeOp->getNumOperands();
        if (isa<fir::ShapeShiftOp>(shapeOp))
          numArgs /= 2;

        // Since the pre-cg rewrite pass requires the shape to be defined by one
        // of fir.shape, fir.shapeshift or fir.shift, we need to create one of
        // these.
        llvm::SmallVector<Value> extents(numArgs, zero);
        auto newShape = builder.create<fir::ShapeOp>(shape.getLoc(), extents);
        declareOp.getShapeMutable().assign(newShape);
      }

      for (OpOperand &typeParam : declareOp.getTypeparamsMutable())
        typeParam.assign(zero);

      declareOp.getDummyScopeMutable().clear();
    }

    // We don't actually need the proper local allocations, but rather maintain
    // the basic form of map operands. We create 1-bit placeholder allocas
    // that we "typecast" to the expected pointer type and replace all uses.
    // Using fir.undefined here instead is not possible because these variables
    // cannot be constants, as that would trigger different codegen for target
    // regions.
    for (Value value : rewriteValues) {
      Location loc = value.getLoc();
      Value placeholder =
          builder.create<fir::AllocaOp>(loc, builder.getI1Type());
      value.replaceAllUsesWith(
          builder.create<fir::ConvertOp>(loc, value.getType(), placeholder));
    }

    // Move omp.map.info dependencies.
    for (hlfir::DeclareOp declareOp : declareOps)
      declareOp->moveBefore(marker);

    // The box_ref argument of fir.box_offset is expected to be the same value
    // that was passed as var_ptr to the corresponding omp.map.info, so we don't
    // need to move its defining op here.
    for (fir::BoxOffsetOp boxOffset : boxOffsets)
      boxOffset->moveBefore(marker);

    marker->erase();

    // Move mapping information users to the end of the new block.
    for (auto targetOp : targetOps)
      std::visit([&block](auto op) { op->moveBefore(&block, block.end()); },
                 targetOp);

    // Add terminator to the new block.
    builder.setInsertionPointToEnd(&block);
    if (funcOp) {
      llvm::SmallVector<Value> returnValues;
      returnValues.reserve(funcOp.getNumResults());
      for (auto type : funcOp.getResultTypes())
        returnValues.push_back(
            builder.create<fir::UndefOp>(funcOp.getLoc(), type));

      builder.create<func::ReturnOp>(funcOp.getLoc(), returnValues);
    } else {
      builder.create<omp::TerminatorOp>(targetDataOp.getLoc());
    }

    // Replace old (now missing ops) region with the new one and remove the
    // temporary clone.
    region.takeBody(newOp->getRegion(0));
    newOp->erase();
    return success();
  }
};
} // namespace
