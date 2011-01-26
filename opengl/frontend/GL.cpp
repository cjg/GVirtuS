/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <errno.h>
#include <sys/ioctl.h>

#include <iostream>

#include <Frontend.h>
#include <Communicator.h>

#include <GL/glx.h>


using namespace std;

enum ShmType {
    POSIX, VMSHM, NONE
};

static Buffer *mspRoutines;
static bool msRoutinesEmpty;
pthread_mutex_t mMutex = PTHREAD_MUTEX_INITIALIZER;
static ShmType mShmType;
static char *mspShmName;
static char *mspFramebuffer;
static pthread_spinlock_t *mspLock;

static void __attribute__((constructor)) _GL_init(void) {
    string communicator = string(getenv("COMMUNICATOR"));
    const char *shmtype = getenv("SHM");
    if (shmtype == NULL)
        mShmType = NONE;
    else if (strcasecmp(shmtype, "posix") == 0)
        mShmType = POSIX;
    else if (strcasecmp(shmtype, "vmshm") == 0)
        mShmType = VMSHM;
    else {
        cerr << "ShmType not recognized, falling back to 'None'." << endl;
        mShmType = NONE;
    }
    cout << mShmType << endl;
    Communicator *c = Communicator::Get(communicator);
    Frontend::GetFrontend(c);
    mspRoutines = new Buffer();
    msRoutinesEmpty = true;
    if (!XInitThreads()) {
        cout << "Xlib not thread safe\n";
        exit(1);
    }
}

static inline void FlushRoutines() {
    if (msRoutinesEmpty)
        return;
    pthread_mutex_lock(&mMutex);
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    f->Execute("gl__ExecuteRoutines", mspRoutines);
    pthread_mutex_unlock(&mMutex);
    mspRoutines->Reset();
    msRoutinesEmpty = true;
}

static inline Buffer *AddRoutine(const char *routine) {
    mspRoutines->AddString(routine);
    cout << routine << endl;
    msRoutinesEmpty = false;
    return mspRoutines;
}

static inline Frontend *GetFrontend() {
    FlushRoutines();
    pthread_mutex_lock(&mMutex);
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    return f;
}

struct WindowInfo {
    Display *mpDpy;
    GLXDrawable mDrawable;
    int mWidth;
    int mHeight;
};

static void InitFramebuffer() {
    int fd;
    cout << "InitFramebuffer: " << mShmType << endl;
    if (mShmType == NONE) {
        mspFramebuffer = new char[300 * 300 * sizeof (int) ];
        mspLock = NULL;
        return;
    } else if (mShmType == POSIX) {
        fd = shm_open(mspShmName, O_RDWR, S_IRUSR | S_IWUSR);
        if (ftruncate(fd, 300 * 300 * sizeof (int)));
    } else {
        fd = open("/dev/vmshm0", O_RDWR);
        ioctl(fd, 0, mspShmName);
    }
    mspLock = reinterpret_cast<pthread_spinlock_t *> (mmap(NULL,
            300 * 300 * sizeof (int), PROT_READ | PROT_WRITE, MAP_SHARED, fd,
            0));
    mspFramebuffer = ((char *) mspLock) + sizeof(pthread_spinlock_t);
}

static void Lock() {
    if(mspLock)
        pthread_spin_lock(mspLock);
}

static void Unlock() {
    if(mspLock)
        pthread_spin_unlock(mspLock);
}

static inline char *GetFramebuffer() {
    if (mShmType != NONE)
        return mspFramebuffer;
    pthread_mutex_lock(&mMutex);
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    f->Execute("gl__GetBuffer");
    memmove(mspFramebuffer, f->GetOutputBuffer()->GetBuffer(),
            300 * 300 * sizeof (int));
    pthread_mutex_unlock(&mMutex);
    return mspFramebuffer;
}

static void *update(void *__w) {
    WindowInfo *w = (WindowInfo *) __w;
    InitFramebuffer();
    char *buffer;
    int *row = new int[300];
    while (true) {
        Lock();
        buffer = GetFramebuffer();
        for(int i = 0; i < 300 / 2; i++) {
            memmove(row, buffer + (299 - i) * 300 * sizeof(int), 300 * sizeof(int));
            memmove(buffer + (299 - i) * 300 * sizeof(int), buffer + i * 300 * sizeof(int), 300 * sizeof(int));
            memmove(buffer + i * 300 * sizeof(int), row, 300 * sizeof(int));
        }
        XImage *img = XCreateImage(w->mpDpy, CopyFromParent,
                24, ZPixmap, 0, buffer, w->mWidth, w->mHeight, 32,
                w->mWidth * 4);
        img->bits_per_pixel = 32;
        XPutImage(w->mpDpy, w->mDrawable, DefaultGC(w->mpDpy, 0), img, 0, 0, 0,
                0, w->mWidth, w->mHeight);
        Unlock();
        usleep(5000);
    }
    return NULL;
}

extern "C" XVisualInfo* glXChooseVisual(Display *dpy, int screen, int * attribList) {
    Frontend *f = GetFrontend();
    Buffer *in = f->GetInputBuffer();
    in->AddString(XDisplayString(dpy));
    in->Add(screen);
    int n;
    for (n = 0; attribList[n] != None; n++);
    in->Add(attribList, n + 1);
    f->Execute("glXChooseVisual");
    if (f->GetExitCode() != 0)
        return NULL;
    Buffer *out = f->GetOutputBuffer();
    XVisualInfo *vis = out->Get<XVisualInfo > (1);
    vis->visual = out->Get<Visual > (1);
    VisualID id = out->Get<VisualID > ();
    pthread_mutex_unlock(&mMutex);

    XVisualInfo vis_template, *visual;
    vis_template.screen = screen;
    vis_template.visualid = XVisualIDFromVisual(DefaultVisual(dpy, screen));
    Visual *v = DefaultVisual(dpy, screen);
    cout << (void *) v->red_mask << endl;
    return XGetVisualInfo(dpy, VisualScreenMask | VisualIDMask,
            &vis_template, &n);
}

extern "C" GLXContext glXCreateContext(Display *dpy, XVisualInfo *vis,
        GLXContext shareList, Bool direct) {
    Frontend *f = GetFrontend();
    Buffer *in = f->GetInputBuffer();
    //in->AddString(XDisplayString(dpy));
    in->Add(vis);
    in->Add(vis->visual);
    in->Add((uint64_t) shareList);
    in->Add(direct);
    f->Execute("glXCreateContext");
    if (f->GetExitCode() != 0)
        return NULL;
    Buffer *out = f->GetOutputBuffer();
    GLXContext ctx = (GLXContext) out->Get<uint64_t > ();
    pthread_mutex_unlock(&mMutex);
    return ctx;
}

extern "C" Bool glXMakeCurrent(Display *dpy, GLXDrawable drawable,
        GLXContext ctx) {
    Frontend *f = GetFrontend();
    Buffer *in = f->GetInputBuffer();
    //in->AddString(XDisplayString(dpy));
    in->Add(drawable);
    in->Add((uint64_t) ctx);
    bool use_shm = mShmType != NONE;
    cout << "use_shm: " << use_shm << endl;
    in->Add(use_shm);
    f->Execute("glXMakeCurrent");
    Buffer *out = f->GetOutputBuffer();
    WindowInfo *w = new WindowInfo;
    w->mDrawable = drawable;
    w->mpDpy = dpy;
    w->mWidth = 300;
    w->mHeight = 300;
    Bool result = out->Get<Bool > ();
    if (mShmType != NONE)
        mspShmName = strdup(out->AssignString());
    pthread_t tid;
    pthread_create(&tid, NULL, update, w);
    pthread_mutex_unlock(&mMutex);
    return result;
}

extern "C" const char *glXQueryExtensionsString(Display *dpy, int screen) {
    Frontend *f = GetFrontend();
    Buffer *in = f->GetInputBuffer();
    //in->AddString(XDisplayString(dpy));
    in->Add(screen);
    f->Execute("glXQueryExtensionsString");
    Buffer *out = f->GetOutputBuffer();
    char *result = strdup(out->AssignString());
    pthread_mutex_unlock(&mMutex);
    return result;
}

extern "C" Bool glXQueryExtension(Display *dpy, int *errorBase,
    int *eventBase) {
    Frontend *f = GetFrontend();
    f->Execute("glXQueryExtension");
    if(f->GetExitCode() != 0)
        return False;
    if(errorBase)
        *errorBase = f->GetOutputBuffer()->Get<int>();
    if(eventBase)
        *eventBase = f->GetOutputBuffer()->Get<int>();
    return True;
}

extern "C" GLuint glGenLists(GLsizei range) {
    Frontend *f = GetFrontend();
    Buffer *in = f->GetInputBuffer();
    in->Add(range);
    f->Execute("glGenLists");
    GLuint result = f->GetOutputBuffer()->Get<GLuint > ();
    pthread_mutex_unlock(&mMutex);
    return result;
}

extern "C" void glLightfv(GLenum light, GLenum pname, const GLfloat *params) {
    Buffer *in = AddRoutine("glLightfv");
    int n = 0;
    switch (pname) {
        case GL_AMBIENT:
        case GL_DIFFUSE:
        case GL_SPECULAR:
        case GL_POSITION:
            n = 4;
            break;
        case GL_SPOT_DIRECTION:
            n = 3;
            break;
        case GL_SPOT_EXPONENT:
        case GL_SPOT_CUTOFF:
        case GL_CONSTANT_ATTENUATION:
        case GL_LINEAR_ATTENUATION:
        case GL_QUADRATIC_ATTENUATION:
            n = 1;
            break;
    }
    in->Add(light);
    in->Add(pname);
    in->Add(params, n);
}

extern "C" void glEnable(GLenum cap) {
    Buffer *in = AddRoutine("glEnable");
    in->Add(cap);
}

extern "C" void glNewList(GLuint list, GLenum mode) {
    Buffer *in = AddRoutine("glNewList");
    in->Add(list);
    in->Add(mode);
}

extern "C" void glMaterialfv(GLenum face, GLenum pname, const GLfloat *params) {
    Buffer *in = AddRoutine("glMaterialfv");
    int n = 0;
    switch (pname) {
        case GL_AMBIENT:
        case GL_DIFFUSE:
        case GL_SPECULAR:
        case GL_EMISSION:
        case GL_AMBIENT_AND_DIFFUSE:
            n = 4;
            break;
        case GL_COLOR_INDEXES:
            n = 3;
            break;
        case GL_SHININESS:
            n = 1;
            break;
    }
    in->Add(face);
    in->Add(pname);
    in->Add(params, n);
}

extern "C" void glShadeModel(GLenum mode) {
    Buffer *in = AddRoutine("glShadeModel");
    in->Add(mode);
}

extern "C" void glNormal3f(GLfloat nx, GLfloat ny, GLfloat nz) {
    Buffer *in = AddRoutine("glNormal3f");
    in->Add(nx);
    in->Add(ny);
    in->Add(nz);
}

extern "C" void glBegin(GLenum mode) {
    Buffer *in = AddRoutine("glBegin");
    in->Add(mode);
}

extern "C" void glVertex3f(GLfloat x, GLfloat y, GLfloat z) {
    Buffer *in = AddRoutine("glVertex3f");
    in->Add(x);
    in->Add(y);
    in->Add(z);
}

extern "C" void glEnd(void) {
    AddRoutine("glEnd");
}

extern "C" void glEndList(void) {
    AddRoutine("glEndList");
}

extern "C" void glViewport(GLint x, GLint y, GLsizei width, GLsizei height) {
    Buffer *in = AddRoutine("glViewport");
    in->Add(x);
    in->Add(y);
    in->Add(width);
    in->Add(height);
}

extern "C" void glMatrixMode(GLenum mode) {
    Buffer *in = AddRoutine("glMatrixMode");
    in->Add(mode);
}

extern "C" void glLoadIdentity(void) {
    AddRoutine("glLoadIdentity");
}

extern "C" void glFrustum(GLdouble left, GLdouble right, GLdouble bottom,
        GLdouble top, GLdouble nearVal, GLdouble farVal) {
    Buffer *in = AddRoutine("glFrustum");
    in->Add(left);
    in->Add(right);
    in->Add(bottom);
    in->Add(top);
    in->Add(nearVal);
    in->Add(farVal);
}

extern "C" void glTranslatef(GLfloat x, GLfloat y, GLfloat z) {
    Buffer *in = AddRoutine("glTranslatef");
    in->Add(x);
    in->Add(y);
    in->Add(z);
}

extern "C" void glClear(GLbitfield mask) {
    Buffer *in = AddRoutine("glClear");
    in->Add(mask);
}

extern "C" void glCallList(GLuint list) {
    Buffer *in = AddRoutine("glCallList");
    in->Add(list);
}

extern "C" void glPopMatrix(void) {
    AddRoutine("glPopMatrix");
}

extern "C" void glPushMatrix(void) {
    AddRoutine("glPushMatrix");
}

extern "C" void glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z) {
    Buffer *in = AddRoutine("glRotatef");
    in->Add(angle);
    in->Add(x);
    in->Add(y);
    in->Add(z);
}

#include <cstdio>

extern "C" void glXSwapBuffers(Display *dpy, GLXDrawable drawable) {
    Frontend *f = GetFrontend();
    Buffer *in = f->GetInputBuffer();
    //in->AddString(XDisplayString(dpy));
    in->Add(drawable);
    f->Execute("glXSwapBuffers");
    pthread_mutex_unlock(&mMutex);
    //XImage *img = XCreateImage(dpy, CopyFromParent,
    //        24, ZPixmap, 0, GetFramebuffer(), 300, 300, 32,
    //        300 * 4);
    //img->bits_per_pixel = 32;
    //XPutImage(dpy, drawable, DefaultGC(dpy, 0), img, 0, 0, 0,
    //        0, 300, 300);
}

extern "C" GLXFBConfig * glXChooseFBConfig(	Display *  	dpy,
 	int  	screen,
 	const int *  	attrib_list,
 	int *  	nelements) {}

extern "C" XVisualInfo * glXGetVisualFromFBConfig(	Display *  	dpy,
 	GLXFBConfig  	config) {

    int n;
    XVisualInfo vis_template, *visual;
    vis_template.screen = 0;
    vis_template.visualid = XVisualIDFromVisual(DefaultVisual(dpy, 0));

    return XGetVisualInfo(dpy, VisualScreenMask | VisualIDMask,
            &vis_template, &n);

}

extern "C" GLXContext glXCreateNewContext(	Display *  	dpy,
 	GLXFBConfig  	config,
 	int  	render_type,
 	GLXContext  	share_list,
 	Bool  	direct) {
    int n;
    XVisualInfo vis_template, *visual;
    vis_template.screen = 0;
    vis_template.visualid = XVisualIDFromVisual(DefaultVisual(dpy, 0));

    return glXCreateContext(dpy,    XGetVisualInfo(dpy, VisualScreenMask | VisualIDMask,
            &vis_template, &n), share_list, direct);
}