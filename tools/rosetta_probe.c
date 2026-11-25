#include <stdio.h>
#include <signal.h>
#include <setjmp.h>
#include <string.h>
#include <stdint.h>
#include <cpuid.h>
static sigjmp_buf jb; static volatile int sig;
static void h(int s){ sig=s; siglongjmp(jb,1); }
static uint8_t line[256] __attribute__((aligned(64)));
#define T(name, ...) do { sig=0; if(!sigsetjmp(jb,1)){ __VA_ARGS__; printf("%-34s OK\n", name);} else printf("%-34s FAULT (signal %d)\n", name, sig); fflush(stdout);} while(0)
int main(void){
  signal(SIGILL,h); signal(SIGSEGV,h); signal(SIGBUS,h);
  unsigned a,b,c,d; char v[13]={0};
  __cpuid(0,a,b,c,d); memcpy(v,&b,4); memcpy(v+4,&d,4); memcpy(v+8,&c,4);
  printf("cpuid vendor: %s, max leaf %u\n", v, a);
  __cpuid(0x80000001,a,b,c,d);
  printf("cpuid ext: sse4a=%u monitorx=%u (ecx bit29) lzcnt=%u\n", (c>>6)&1, (c>>29)&1, (c>>5)&1);
  __cpuid_count(7,0,a,b,c,d);
  printf("cpuid 7: avx2=%u bmi1=%u bmi2=%u clwb=%u sha=%u\n", (b>>5)&1,(b>>3)&1,(b>>8)&1,(b>>24)&1,(b>>29)&1);
  __cpuid(1,a,b,c,d);
  printf("cpuid 1: avx=%u osxsave=%u fma=%u f16c=%u movbe=%u aes=%u pclmul=%u\n",(c>>28)&1,(c>>27)&1,(c>>12)&1,(c>>29)&1,(c>>22)&1,(c>>25)&1,(c>>1)&1);
  printf("\n");
  T("xgetbv (XCR0)", { unsigned lo,hi; __asm__ volatile("xor %%ecx,%%ecx; xgetbv":"=a"(lo),"=d"(hi)::"ecx"); printf("  XCR0=%#x ", lo); });
  T("monitorx  0F 01 FA", { void*p=line; __asm__ volatile("xor %%ecx,%%ecx; xor %%edx,%%edx; .byte 0x0f,0x01,0xfa"::"a"(p):"ecx","edx","memory"); });
  T("mwaitx    0F 01 FB", { __asm__ volatile("xor %%eax,%%eax; xor %%ecx,%%ecx; xor %%ebx,%%ebx; .byte 0x0f,0x01,0xfb":::"eax","ecx","ebx","memory"); });
  T("clzero    0F 01 FC", { void*p=line+64; memset(line,0xAA,sizeof line); __asm__ volatile(".byte 0x0f,0x01,0xfc"::"a"(p):"memory"); printf("  zeroed=%d ", line[64]==0&&line[127]==0&&line[128]==0xAA); });
  T("insertq xmm0,xmm1,8,0 (SSE4a)", { __asm__ volatile(".byte 0xf2,0x0f,0x78,0xc1,0x08,0x00":::"xmm0","xmm1"); });
  T("extrq xmm0,8,0 (SSE4a)", { __asm__ volatile(".byte 0x66,0x0f,0x78,0xc0,0x08,0x00":::"xmm0"); });
  T("clwb", { __asm__ volatile(".byte 0x66,0x0f,0xae,0x30"::"a"(line):"memory"); });
  T("prefetchw", { __asm__ volatile("prefetchw (%0)"::"r"(line)); });
  T("movbe", { uint32_t x=0x11223344,y; __asm__ volatile("movbe %1,%0":"=r"(y):"m"(x)); printf("  %#x ", y); });
  T("lzcnt/tzcnt", { uint32_t x=0x100,l,t; __asm__ volatile("lzcnt %2,%0; tzcnt %2,%1":"=r"(l),"=r"(t):"r"(x)); printf("  %u/%u ", l,t); });
  T("andn/bextr (BMI1)", { uint32_t r; __asm__ volatile("andn %2,%1,%0":"=r"(r):"r"(0xF0u),"r"(0xFFu)); printf("  %#x ", r); });
  T("pext/pdep/shlx/mulx (BMI2)", { uint64_t r,m=0xFF00,s=0xAB00; __asm__ volatile("pext %2,%1,%0":"=r"(r):"r"(s),"r"(m)); printf("  pext=%#llx ", (unsigned long long)r); });
  T("vcvtph2ps (F16C)", { float f[4]; uint16_t hh[4]={0x3c00,0x4000,0,0}; __asm__ volatile("vcvtph2ps (%1),%%xmm0; vmovups %%xmm0,(%0)"::"r"(f),"r"(hh):"xmm0","memory"); printf("  %g %g ", f[0], f[1]); });
  T("aesenc + pclmulqdq", { __asm__ volatile("aesenc %%xmm1,%%xmm0; pclmulqdq $0,%%xmm1,%%xmm0":::"xmm0","xmm1"); });
  T("vfmadd231ps ymm (FMA, 256-bit)", { __asm__ volatile("vfmadd231ps %%ymm1,%%ymm2,%%ymm0; vzeroupper":::"xmm0","xmm1","xmm2"); });
  T("rdtsc / rdtscp", { unsigned lo,hi,aux; __asm__ volatile("rdtsc":"=a"(lo),"=d"(hi)); __asm__ volatile("rdtscp":"=a"(lo),"=d"(hi),"=c"(aux)); });
  return 0;
}
