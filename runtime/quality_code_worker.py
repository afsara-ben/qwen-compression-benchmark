"""Fresh-process HumanEval execution with Linux filesystem and syscall isolation."""
import ctypes
import errno
import json
import os
import resource
import signal
import sys


def isolate(directory):
    libc = ctypes.CDLL(None, use_errno=True)
    sec = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError('Landlock unavailable')
    # ABI 1 filesystem operations, plus cross-directory REFER on ABI >=2.
    handled = (1 << (14 if abi >= 2 else 13)) - 1
    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]
    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]
    attr = Ruleset(handled)
    ruleset = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), 'create Landlock ruleset')
    def allow(path, access):
        if not os.path.exists(path): return
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        rule = PathRule(access, fd)
        rc = libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0)
        os.close(fd)
        if rc: raise OSError(ctypes.get_errno(), 'add Landlock rule')
    for path in {os.path.realpath(sys.base_prefix),'/usr/lib','/lib','/lib64'}:
        allow(path, (1 << 2) | (1 << 3))  # Read file/directory, no execution.
    allow('/dev/null', (1 << 1) | (1 << 2))
    allow('/dev/urandom', 1 << 2)
    allow(directory, handled & ~1)  # Per-test temporary files, no executable files.
    if libc.prctl(38, 1, 0, 0, 0): raise OSError('no_new_privs')
    if libc.syscall(446, ruleset, 0): raise OSError(ctypes.get_errno(), 'restrict Landlock')
    os.close(ruleset)
    sec.seccomp_init.argtypes=[ctypes.c_uint32]; sec.seccomp_init.restype=ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]; sec.seccomp_syscall_resolve_name.restype=ctypes.c_int
    sec.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    sec.seccomp_load.argtypes=[ctypes.c_void_p]
    ctx=sec.seccomp_init(0x7fff0000)
    if not ctx: raise RuntimeError('seccomp_init failed')
    denied = ['socket','socketpair','connect','bind','listen','accept','accept4',
              'sendto','sendmsg','sendmmsg','recvfrom','recvmsg','recvmmsg',
              'clone','clone3','fork','vfork','execve','execveat','ptrace',
              'process_vm_readv','process_vm_writev','kill','tkill','tgkill',
              'mount','umount2','unshare','setns','bpf','keyctl','perf_event_open',
              'io_uring_setup','open_by_handle_at','name_to_handle_at']
    for name in denied:
        nr=sec.seccomp_syscall_resolve_name(name.encode())
        if nr >= 0 and sec.seccomp_rule_add(ctx,0x00050000|errno.EPERM,nr,0):
            raise RuntimeError('seccomp rule '+name)
    if sec.seccomp_load(ctx): raise RuntimeError('seccomp_load failed')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(4,4))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024**2,1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))


def main():
    payload=json.load(sys.stdin)
    try:
        isolate(os.getcwd())
    except BaseException:
        os._exit(120)  # Infrastructure failure must never be a model score.
    def timeout(*args): raise TimeoutError('HumanEval 3 second execution limit')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 3)
    try:
        exec(compile(payload['program'],'<candidate>','exec'), {'__name__':'__main__'})
    except BaseException:
        os._exit(1)
    os._exit(0)


if __name__ == '__main__': main()
