#include "openzl/zl_codec_output_cache.h"
#include <cstdlib>

// cdbench <iters> <elt_bytes> <chunk_list> <variant>...   variant = path/to/model.zlc | zstd:<level> | tf:<elt_bytes>[x<streams>] (upstream Compression Transformer, numeric input; x3 = split interleaved xyz into 3 streams)
// Per chunk, every variant compresses and decompresses back to back (order alternates per iteration).
// A model that fails to compress a chunk falls back to zstd-1 for that chunk (counted in the last column).
// Prints one row per chunk directory (= source file) and variant: dir, variant, src bytes, compressed bytes, sum of min compress ns, sum of min decompress ns, fallback count.
#include "openzl/openzl.h"
#include "openzl/codecs/zl_brute_force_selector.h"
#include "openzl/zl_compressor_serialization.h"
#include <zstd.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
static std::string rd(const std::string& p){std::ifstream f(p,std::ios::binary);std::stringstream s;s<<f.rdbuf();return s.str();}
using clk=std::chrono::steady_clock;
static double ns(clk::time_point a, clk::time_point b){return std::chrono::duration<double,std::nano>(b-a).count();}
struct V { std::string name; int zlevel=0; ZL_Compressor* comp=nullptr; ZL_CCtx* cctx=nullptr; int tfw=0, tfs=1; };
int main(int argc,char**argv){
  if(argc<5){fprintf(stderr,"usage\n");return 1;}
  int iters=atoi(argv[1]); size_t elt=strtoul(argv[2],0,10);
  std::vector<std::string> chunks, cdir; {std::ifstream L(argv[3]); std::string p; while(std::getline(L,p)) if(!p.empty()){ chunks.push_back(rd(p)); cdir.push_back(p.substr(0,p.rfind('/'))); }}
  std::vector<V> vs;
  for(int a=4;a<argc;++a){ V v; v.name=argv[a];
    if(v.name.rfind("zstd:",0)==0){ v.zlevel=atoi(v.name.c_str()+5); }
    else if(v.name.rfind("tf:",0)==0){ v.tfw=atoi(v.name.c_str()+3); const char* x=strchr(v.name.c_str(),'x'); if(x) v.tfs=atoi(x+1);
      v.comp=ZL_Compressor_create(); ZL_Compressor_setParameter(v.comp,ZL_CParam_formatVersion,ZL_MAX_FORMAT_VERSION);
      if(ZL_isError(ZL_Compressor_selectStartingGraphID(v.comp,ZL_GRAPH_TRANSFORMER_NUMERIC))){fprintf(stderr,"tf graph\n");return 1;}
      v.cctx=ZL_CCtx_create(); ZL_CCtx_setParameter(v.cctx,ZL_CParam_stickyParameters,1); ZL_CCtx_refCompressor(v.cctx,v.comp); }
    else { std::string s=rd(v.name); v.comp=ZL_Compressor_create(); ZL_CompressorDeserializer* d=ZL_CompressorDeserializer_create();
      ZL_Report r=ZL_CompressorDeserializer_deserialize(d,v.comp,s.data(),s.size(),nullptr,0);
      if(ZL_isError(r)){fprintf(stderr,"deserialize failed: %s\n",v.name.c_str());return 1;}
      ZL_CompressorDeserializer_free(d);
      v.cctx=ZL_CCtx_create(); ZL_CCtx_setParameter(v.cctx,ZL_CParam_formatVersion,ZL_MAX_FORMAT_VERSION);
      ZL_CCtx_setParameter(v.cctx,ZL_CParam_stickyParameters,1);
      if(getenv("CDB_TRY_CACHE")) ZL_CCtx_setTryGraphCacheBudget(v.cctx,(size_t)atol(getenv("CDB_TRY_CACHE"))<<20);
      ZL_CCtx_refCompressor(v.cctx,v.comp); }
    vs.push_back(v); }
  size_t nv=vs.size(); ZL_DCtx* dctx=ZL_DCtx_create(); ZSTD_CCtx* zc=ZSTD_createCCtx(); ZSTD_DCtx* zd=ZSTD_createDCtx();
  struct Acc { size_t sb=0, cb=0, fails=0; double ct=0, dt=0; };
  std::map<std::string, std::vector<Acc>> per; // chunk dir -> per-variant sums
  std::vector<char> dst, out, tfbuf;
  for(size_t ci=0; ci<chunks.size(); ++ci){ auto& c=chunks[ci];
    if(ci==0 || cdir[ci]!=cdir[ci-1]) for(auto& v: vs) if(v.cctx) ZL_CCtx_resetBruteForceNarrowing(v.cctx); auto& A=per.try_emplace(cdir[ci], nv).first->second; size_t cap=std::max(ZL_compressBound(c.size()),ZSTD_compressBound(c.size()))*2+4096; dst.resize(cap); out.resize(c.size()+64);
    std::vector<double> bc(nv,1e30), bd(nv,1e30);
    for(int it=0;it<iters;++it) for(size_t k=0;k<nv;++k){ size_t i=(it%2)?nv-1-k:k; V& v=vs[i]; size_t csz;
      bool fb=false; auto t0=clk::now();
      if(v.tfw){ size_t n=c.size()/v.tfw/v.tfs; std::vector<const ZL_TypedRef*> refs;
        if(v.tfs==1) refs.push_back(ZL_TypedRef_createNumeric(c.data(),v.tfw,n));
        else { tfbuf.resize(c.size()); for(int s=0;s<v.tfs;++s) for(size_t j=0;j<n;++j) memcpy(&tfbuf[(s*n+j)*v.tfw],&c[(j*v.tfs+s)*v.tfw],v.tfw);
          for(int s=0;s<v.tfs;++s) refs.push_back(ZL_TypedRef_createNumeric(&tfbuf[s*n*v.tfw],v.tfw,n)); }
        ZL_Report r=ZL_CCtx_compressMultiTypedRef(v.cctx,dst.data(),cap,refs.data(),refs.size());
        for(auto* t: refs) ZL_TypedRef_free((ZL_TypedRef*)t);
        if(ZL_isError(r)){ A[i].fails+=(it==0); fb=true; } else csz=ZL_validResult(r); }
      else if(v.cctx){ ZL_Report r=ZL_CCtx_compress(v.cctx,dst.data(),cap,c.data(),c.size()); if(ZL_isError(r)){ A[i].fails+=(it==0); fb=true; } else csz=ZL_validResult(r);} 
      if(!v.cctx || fb) { int lvl=fb?1:v.zlevel; csz=ZSTD_compressCCtx(zc,dst.data(),cap,c.data(),c.size(),lvl); if(ZSTD_isError(csz)){fprintf(stderr,"zstd err\n");return 1;} }
      auto t1=clk::now(); size_t dsz; clk::time_point t2tf;
      if(v.tfw && !fb){ size_t n=c.size()/v.tfw/v.tfs; std::vector<ZL_TypedBuffer*> tb(v.tfs); for(auto& b: tb) b=ZL_TypedBuffer_create();
        ZL_Report r=ZL_DCtx_decompressMultiTBuffer(dctx,tb.data(),tb.size(),dst.data(),csz); if(ZL_isError(r)){fprintf(stderr,"tf decompress error\n");return 1;}
        t2tf=clk::now(); dsz=0;
        for(int s=0;s<v.tfs;++s){ const char* p=(const char*)ZL_TypedBuffer_rPtr(tb[s]); size_t ne=ZL_TypedBuffer_numElts(tb[s]);
          for(size_t j=0;j<ne;++j) memcpy(&out[(j*v.tfs+s)*v.tfw],p+j*v.tfw,v.tfw); dsz+=ne*v.tfw; ZL_TypedBuffer_free(tb[s]); } }
      else if(v.cctx && !fb){ ZL_Report r=ZL_DCtx_decompress(dctx,out.data(),out.size(),dst.data(),csz); if(ZL_isError(r)){fprintf(stderr,"decompress error %s\n",v.name.c_str());return 1;} dsz=ZL_validResult(r);} 
      else dsz=ZSTD_decompressDCtx(zd,out.data(),out.size(),dst.data(),csz);
      auto t2=clk::now(); if(v.tfw && !fb) t2=t2tf;
      if(dsz!=c.size()||memcmp(out.data(),c.data(),c.size())){fprintf(stderr,"MISMATCH %s\n",v.name.c_str());return 1;}
      bc[i]=std::min(bc[i],ns(t0,t1)); bd[i]=std::min(bd[i],ns(t1,t2)); if(it==0) A[i].cb+=csz; }
    for(size_t i=0;i<nv;++i){A[i].ct+=bc[i]; A[i].dt+=bd[i]; A[i].sb+=c.size();}
  }
  for(auto& [dir,A]: per) for(size_t i=0;i<nv;++i) printf("%s,%s,%zu,%zu,%.0f,%.0f,%zu\n",dir.c_str(),vs[i].name.c_str(),A[i].sb,A[i].cb,A[i].ct,A[i].dt,A[i].fails);
  return 0;
}
