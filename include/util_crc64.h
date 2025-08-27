#ifndef JV_UTIL_CRC64_H
#define JV_UTIL_CRC64_H
#include <stddef.h>
#include <stdint.h>
uint64_t jv_crc64(const void* data, size_t len);
#endif
