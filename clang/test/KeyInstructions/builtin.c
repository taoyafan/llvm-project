
// RUN: %clang -gkey-instructions -x c++ %s -gmlt -gno-column-info -S -emit-llvm -o - -ftrivial-auto-var-init=zero \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang -gkey-instructions -x c %s -gmlt -gno-column-info -S -emit-llvm -o - -ftrivial-auto-var-init=zero \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

void fun() {
// CHECK: %a = alloca ptr, align 8
// CHECK: %0 = alloca i8, i64 4{{.*}}, !dbg [[G1R2:!.*]]
// CHECK: call void @llvm.memset{{.*}}, !dbg [[G1R1:!.*]], !annotation
// CHECK: store ptr %0, ptr %a{{.*}}, !dbg [[G1R1:!.*]]
    void *a = __builtin_alloca(4);

// CHECK: %1 = alloca i8, i64 4{{.*}}, !dbg [[G2R2:!.*]]
// CHECK: call void @llvm.memset{{.*}}, !dbg [[G2R1:!.*]], !annotation
// CHECK: store ptr %1, ptr %b{{.*}}, !dbg [[G2R1:!.*]]
    void *b = __builtin_alloca_with_align(4, 8);
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
