#include "util_crc64.h"
#define POLY 0x42F0E1EBA9EA3693ULL
static uint64_t table[256];
static int      initd = 0;
static void init(void){
  if(initd) return;
  for (unsigned i=0; i<256; ++i){
    uint64_t crc = (uint64_t)i << 56;
    for (int j=0; j<8; ++j)
      crc = (crc & 0x8000000000000000ULL) ? (crc << 1) ^ POLY : (crc << 1);
    table[i]=crc;
  }
  initd=1;
}
uint64_t jv_crc64(const void* data, size_t len){
  init();
  const unsigned char* p = (const unsigned char*)data;
  uint64_t crc = 0ULL;
  for (size_t i=0;i<len;++i){
    unsigned idx = (unsigned)((crc >> 56) ^ p[i]) & 0xFFU;
    crc = table[idx] ^ (crc << 8);
  }
  return crc;
}
