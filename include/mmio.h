#ifndef JV_MMIO_H
#define JV_MMIO_H
#include <stdint.h>
#include <stddef.h>

typedef struct {
  void*   addr;
  size_t  size;
#ifdef _WIN32
  void*   hFile;
  void*   hMap;
#endif
} jv_mmap_t;

int  jv_mmap_read (const char* path, jv_mmap_t* out);
int  jv_mmap_rw   (const char* path, size_t size, jv_mmap_t* out); // create/resize then map
int  jv_munmap    (jv_mmap_t* m);
int  jv_msync     (jv_mmap_t* m);

#endif
