//===--- CIRGenCall.cpp - Encapsulate calling convention details ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function definition used
// to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCall.h"
#include "CIRGenFunction.h"
#include "CIRGenFunctionInfo.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

CIRGenFunctionInfo *
CIRGenFunctionInfo::create(CanQualType resultType,
                           llvm::ArrayRef<CanQualType> argTypes,
                           RequiredArgs required) {
  void *buffer = operator new(totalSizeToAlloc<ArgInfo>(argTypes.size() + 1));

  CIRGenFunctionInfo *fi = new (buffer) CIRGenFunctionInfo();

  fi->required = required;
  fi->numArgs = argTypes.size();
  fi->getArgsBuffer()[0].type = resultType;
  for (unsigned i = 0; i < argTypes.size(); ++i)
    fi->getArgsBuffer()[i + 1].type = argTypes[i];

  return fi;
}

cir::FuncType CIRGenTypes::getFunctionType(const CIRGenFunctionInfo &fi) {
  bool inserted = functionsBeingProcessed.insert(&fi).second;
  assert(inserted && "Recursively being processed?");

  mlir::Type resultType = nullptr;
  const cir::ABIArgInfo &retAI = fi.getReturnInfo();

  switch (retAI.getKind()) {
  case cir::ABIArgInfo::Ignore:
    // TODO(CIR): This should probably be the None type from the builtin
    // dialect.
    resultType = nullptr;
    break;

  case cir::ABIArgInfo::Direct:
    resultType = retAI.getCoerceToType();
    break;

  default:
    assert(false && "NYI");
  }

  SmallVector<mlir::Type, 8> argTypes;
  unsigned argNo = 0;
  CIRGenFunctionInfo::const_arg_iterator it = fi.arg_begin(),
                                         ie = it + fi.getNumRequiredArgs();
  for (; it != ie; ++it, ++argNo) {
    const auto &argInfo = it->info;

    switch (argInfo.getKind()) {
    default:
      llvm_unreachable("NYI");
    case cir::ABIArgInfo::Direct:
      mlir::Type argType = argInfo.getCoerceToType();
      argTypes.push_back(argType);
      break;
    }
  }

  bool erased = functionsBeingProcessed.erase(&fi);
  assert(erased && "Not in set?");

  return cir::FuncType::get(argTypes,
                            (resultType ? resultType : builder.getVoidTy()),
                            fi.isVariadic());
}

CIRGenCallee CIRGenCallee::prepareConcreteCallee(CIRGenFunction &cgf) const {
  assert(!cir::MissingFeatures::opCallVirtual());
  return *this;
}

static const CIRGenFunctionInfo &
arrangeFreeFunctionLikeCall(CIRGenTypes &cgt, CIRGenModule &cgm,
                            const FunctionType *fnType) {

  RequiredArgs required = RequiredArgs::All;

  if (const auto *proto = dyn_cast<FunctionProtoType>(fnType)) {
    if (proto->isVariadic())
      cgm.errorNYI("call to variadic function");
    if (proto->hasExtParameterInfos())
      cgm.errorNYI("call to functions with extra parameter info");
  } else if (cgm.getTargetCIRGenInfo().isNoProtoCallVariadic(
                 cast<FunctionNoProtoType>(fnType)))
    cgm.errorNYI("call to function without a prototype");

  assert(!cir::MissingFeatures::opCallArgs());

  CanQualType retType = fnType->getReturnType()
                            ->getCanonicalTypeUnqualified()
                            .getUnqualifiedType();
  return cgt.arrangeCIRFunctionInfo(retType, {}, required);
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeFreeFunctionCall(const FunctionType *fnType) {
  return arrangeFreeFunctionLikeCall(*this, cgm, fnType);
}

static cir::CIRCallOpInterface emitCallLikeOp(CIRGenFunction &cgf,
                                              mlir::Location callLoc,
                                              cir::FuncOp directFuncOp) {
  CIRGenBuilderTy &builder = cgf.getBuilder();

  assert(!cir::MissingFeatures::opCallSurroundingTry());
  assert(!cir::MissingFeatures::invokeOp());

  assert(builder.getInsertionBlock() && "expected valid basic block");
  assert(!cir::MissingFeatures::opCallIndirect());

  return builder.createCallOp(callLoc, directFuncOp);
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeFreeFunctionType(CanQual<FunctionProtoType> fpt) {
  SmallVector<CanQualType, 16> argTypes;
  for (unsigned i = 0, e = fpt->getNumParams(); i != e; ++i)
    argTypes.push_back(fpt->getParamType(i));
  RequiredArgs required = RequiredArgs::forPrototypePlus(fpt);

  CanQualType resultType = fpt->getReturnType().getUnqualifiedType();
  return arrangeCIRFunctionInfo(resultType, argTypes, required);
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeFreeFunctionType(CanQual<FunctionNoProtoType> fnpt) {
  CanQualType resultType = fnpt->getReturnType().getUnqualifiedType();
  return arrangeCIRFunctionInfo(resultType, {}, RequiredArgs(0));
}

RValue CIRGenFunction::emitCall(const CIRGenFunctionInfo &funcInfo,
                                const CIRGenCallee &callee,
                                ReturnValueSlot returnValue,
                                cir::CIRCallOpInterface *callOp,
                                mlir::Location loc) {
  QualType retTy = funcInfo.getReturnType();
  const cir::ABIArgInfo &retInfo = funcInfo.getReturnInfo();

  assert(!cir::MissingFeatures::opCallArgs());
  assert(!cir::MissingFeatures::emitLifetimeMarkers());

  const CIRGenCallee &concreteCallee = callee.prepareConcreteCallee(*this);
  mlir::Operation *calleePtr = concreteCallee.getFunctionPointer();

  assert(!cir::MissingFeatures::opCallInAlloca());

  mlir::NamedAttrList attrs;
  StringRef funcName;
  if (auto calleeFuncOp = dyn_cast<cir::FuncOp>(calleePtr))
    funcName = calleeFuncOp.getName();

  assert(!cir::MissingFeatures::opCallCallConv());
  assert(!cir::MissingFeatures::opCallSideEffect());
  assert(!cir::MissingFeatures::opCallAttrs());

  assert(!cir::MissingFeatures::invokeOp());

  auto directFuncOp = dyn_cast<cir::FuncOp>(calleePtr);
  assert(!cir::MissingFeatures::opCallIndirect());
  assert(!cir::MissingFeatures::opCallAttrs());

  cir::CIRCallOpInterface theCall = emitCallLikeOp(*this, loc, directFuncOp);

  if (callOp)
    *callOp = theCall;

  assert(!cir::MissingFeatures::opCallMustTail());
  assert(!cir::MissingFeatures::opCallReturn());

  RValue ret;
  switch (retInfo.getKind()) {
  case cir::ABIArgInfo::Direct: {
    mlir::Type retCIRTy = convertType(retTy);
    if (retInfo.getCoerceToType() == retCIRTy &&
        retInfo.getDirectOffset() == 0) {
      switch (getEvaluationKind(retTy)) {
      case cir::TEK_Scalar: {
        mlir::ResultRange results = theCall->getOpResults();
        assert(results.size() == 1 && "unexpected number of returns");

        // If the argument doesn't match, perform a bitcast to coerce it. This
        // can happen due to trivial type mismatches.
        if (results[0].getType() != retCIRTy)
          cgm.errorNYI(loc, "bitcast on function return value");

        mlir::Region *region = builder.getBlock()->getParent();
        if (region != theCall->getParentRegion())
          cgm.errorNYI(loc, "function calls with cleanup");

        return RValue::get(results[0]);
      }
      default:
        cgm.errorNYI(loc,
                     "unsupported evaluation kind of function call result");
      }
    } else
      cgm.errorNYI(loc, "unsupported function call form");

    break;
  }
  case cir::ABIArgInfo::Ignore:
    // If we are ignoring an argument that had a result, make sure to construct
    // the appropriate return value for our caller.
    ret = getUndefRValue(retTy);
    break;
  default:
    cgm.errorNYI(loc, "unsupported return value information");
  }

  return ret;
}
