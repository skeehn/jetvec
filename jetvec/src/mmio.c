#include "mmio.h"
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
int jv_mmap_read(const char* path, jv_mmap_t* out) {
  memset(out, 0, sizeof(*out));
  HANDLE hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hf == INVALID_HANDLE_VALUE) return -1;
  LARGE_INTEGER sz; if (!GetFileSizeEx(hf, &sz)) { CloseHandle(hf); return -1; }
  HANDLE hm = CreateFileMappingA(hf, NULL, PAGE_READONLY, 0, 0, NULL);
  if (!hm) { CloseHandle(hf); return -1; }
  void* addr = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
  if (!addr) { CloseHandle(hm); CloseHandle(hf); return -1; }
  out->addr = addr; out->size = (size_t)sz.QuadPart; out->hFile = hf; out->hMap = hm; return 0;
}
int jv_mmap_rw(const char* path, size_t size, jv_mmap_t* out) {
  memset(out, 0, sizeof(*out));
  HANDLE hf = CreateFileA(path, GENERIC_READ|GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hf == INVALID_HANDLE_VALUE) return -1;
  LARGE_INTEGER sz; sz.QuadPart = (LONGLONG)size;
  if (!SetFilePointerEx(hf, sz, NULL, FILE_BEGIN) || !SetEndOfFile(hf)) { CloseHandle(hf); return -1; }
  HANDLE hm = CreateFileMappingA(hf, NULL, PAGE_READWRITE, 0, 0, NULL);
  if (!hm) { CloseHandle(hf); return -1; }
  void* addr = MapViewOfFile(hm, FILE_MAP_WRITE|FILE_MAP_READ, 0, 0, size);
  if (!addr) { CloseHandle(hm); CloseHandle(hf); return -1; }
  out->addr = addr; out->size = size; out->hFile = hf; out->hMap = hm; return 0;
}
int jv_munmap(jv_mmap_t* m) {
  if (!m || !m->addr) return 0;
  UnmapViewOfFile(m->addr);
  if (m->hMap) CloseHandle((HANDLE)m->hMap);
  if (m->hFile) CloseHandle((HANDLE)m->hFile);
  memset(m,0,sizeof(*m));
  return 0;
}
int jv_msync(jv_mmap_t* m) {
  if (!m || !m->addr) return -1;
  return FlushViewOfFile(m->addr, 0) ? 0 : -1;
}
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int jv_mmap_read(const char* path, jv_mmap_t* out) {
  memset(out, 0, sizeof(*out));
  int fd = open(path, O_RDONLY);
  if (fd < 0) return -1;
  struct stat st; if (fstat(fd, &st) < 0) { close(fd); return -1; }
  void* addr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) { close(fd); return -1; }
  out->addr = addr; out->size = (size_t)st.st_size;
  close(fd);
  return 0;
}
int jv_mmap_rw(const char* path, size_t size, jv_mmap_t* out) {
  memset(out, 0, sizeof(*out));
  int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return -1;
  if (ftruncate(fd, (off_t)size) < 0) { close(fd); return -1; }
  void* addr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) { close(fd); return -1; }
  out->addr = addr; out->size = size;
  close(fd);
  return 0;
}
int jv_munmap(jv_mmap_t* m) {
  if (!m || !m->addr) return 0;
  munmap(m->addr, m->size);
  memset(m,0,sizeof(*m));
  return 0;
}
int jv_msync(jv_mmap_t* m) {
  if (!m || !m->addr) return -1;
  return msync(m->addr, m->size, MS_SYNC);
}
#endif
