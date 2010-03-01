/* 
 * File:   Subprocess.h
 * Author: cjg
 *
 * Created on December 23, 2009, 8:56 PM
 */

#ifndef _SUBPROCESS_H
#define	_SUBPROCESS_H

#include <sys/types.h>
#include <unistd.h>

class Subprocess {
public:
    Subprocess();
    virtual ~Subprocess();
    int Start(void * arg);
    void Wait();
protected:
    int Run(void * arg);
    static void * EntryPoint(void *);
    virtual void Setup() = 0;
    virtual void Execute(void *) = 0;
    pid_t GetPid();
    void * Arg() const {return mpArg;}
    void Arg(void* a){mpArg = a;}
private:
    pid_t mPid;
    void * mpArg;
};

#endif	/* _SUBPROCESS_H */

