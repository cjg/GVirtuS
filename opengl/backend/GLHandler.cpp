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

/**
 * @file   Backend.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sat Oct 10 10:51:58 2009
 *
 * @brief
 *
 *
 */

#include "GLHandler.h"

#include <cstring>

#include <cuda_runtime_api.h>

#include "CudaUtil.h"

using namespace std;

map<string, GLHandler::GLRoutineHandler> *GLHandler::mspHandlers = NULL;

GLHandler::GLHandler() {
    Initialize();
    mpFramebuffer = NULL;
}

GLHandler::~GLHandler() {

}

Result * GLHandler::Execute(std::string routine, Buffer * input_buffer) {
    map<string, GLHandler::GLRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        throw "No handler for '" + routine + "' found!";
    return it->second(this, input_buffer);
}

const char *GLHandler::InitFramebuffer(size_t size, bool use_shm) {
    if(!use_shm) {
        mpFramebuffer = new char[size];
        return NULL;
    }
    char *name = new char[1024];
    snprintf(name, 1024, "/gvirtus-%d", getpid());
    shm_unlink(name);

    size += sizeof(pthread_spinlock_t);
    int fd = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (ftruncate(fd, size) == -1);

    mpLock = reinterpret_cast<pthread_spinlock_t *> (mmap(NULL, size,
            PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));

    mpFramebuffer = ((char *) mpLock) + sizeof(pthread_spinlock_t);

    pthread_spin_init(mpLock, PTHREAD_PROCESS_SHARED);
    return name;
}

char *GLHandler::GetFramebuffer() {
    return mpFramebuffer;
}

void GLHandler::Initialize() {
    if (mspHandlers != NULL)
        return;
    mspHandlers = new map<string, GLHandler::GLRoutineHandler > ();

    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(XChooseVisual));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(XCreateContext));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(XMakeCurrent));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(XQueryExtensionsString));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(XQueryExtension));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(GenLists));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(NewList));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(ShadeModel));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(Normal3f));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(Begin));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(Vertex3f));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(End));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(EndList));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(Viewport));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(MatrixMode));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(LoadIdentity));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(Frustum));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(Translatef));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(CallList));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(Clear));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(PopMatrix));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(PushMatrix));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(Rotatef));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(XSwapBuffers));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(Enable));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(Lightfv));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(Materialfv));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(__ExecuteRoutines));
    mspHandlers->insert(GL_ROUTINE_HANDLER_PAIR(__GetBuffer));
}

#include <X11/Xlib.h>
#include <GL/glx.h>

static Display *dpy = NULL;
static XVisualInfo *info = NULL;

GL_ROUTINE_HANDLER(XChooseVisual) {
    //dpy = XOpenDisplay(in->AssignString());
    in->AssignString();
    dpy = XOpenDisplay(NULL);
    cout << dpy << endl;
    int screen = in->Get<int>();
    int *attribList = in->AssignAll<int>();
    info = glXChooseVisual(dpy, screen, attribList);
    cout << attribList << endl;
    cout << screen << endl;
    for(int i = 0; attribList[i] != None; i++)
        cout << attribList[i] << endl;
    if(info == NULL)
        return new Result(-1);
    cout << XVisualIDFromVisual(info->visual) << endl;
    Buffer *out = new Buffer();
    out->Add(info->visual);
    out->Add(info);
    out->Add(XVisualIDFromVisual(info->visual));
    return new Result(0, out);
}

GL_ROUTINE_HANDLER(XCreateContext) {
    cout << "cretecontex" << endl;
    //Display *dpy = XOpenDisplay(in->AssignString());
    if(dpy == NULL)
          dpy = XOpenDisplay(NULL);
    int n;
    XVisualInfo vis_template, *visual;
    vis_template.screen = 0;
    vis_template.visualid = XVisualIDFromVisual(DefaultVisual(dpy, 0));

    info = XGetVisualInfo(dpy, VisualScreenMask | VisualIDMask,
            &vis_template, &n);

    in->Get<Visual>(1);
    GLXContext shareList = (GLXContext) in->Get<uint64_t>();
    Bool direct = in->Get<Bool>();
    GLXContext ctx = glXCreateContext(dpy, info, shareList, direct);
    if(ctx == NULL)
        return new Result(-1);
    Buffer *out = new Buffer();
    out->Add((uint64_t) ctx);
    return new Result(0, out);
}
GLXDrawable drawable;
GLXDrawable drawable2;
GL_ROUTINE_HANDLER(XMakeCurrent) {
    cout << "makecurrent" << endl;
    //Display *dpy = XOpenDisplay(in->AssignString());
    drawable = in->Get<GLXDrawable>();
    GLXContext ctx = (GLXContext) in->Get<uint64_t>();
    bool use_shm = in->Get<bool>();

    drawable = XCreateSimpleWindow(dpy, XRootWindow(dpy, DefaultScreen(dpy)), 0, 0, 300, 300, 0, 0, 0);

    const char *name = pThis->InitFramebuffer(300 * 300 * sizeof(int), use_shm);
    XMapWindow(dpy, drawable);
    //XMapWindow(dpy, drawable2);
    Bool result = glXMakeCurrent(dpy, drawable, ctx);
    Buffer *out = new Buffer();
    out->Add(result);
    if(use_shm)
        out->AddString(name);
    return new Result((result == false ? -1 : 0), out);
}

GL_ROUTINE_HANDLER(XQueryExtensionsString) {
    //Display *dpy = XOpenDisplay(in->AssignString());
    int screen = in->Get<int>();
    const char *result = glXQueryExtensionsString(dpy, screen);
    Buffer *out = new Buffer();
    out->AddString(result);
    return new Result(0, out);
}

GL_ROUTINE_HANDLER(GenLists) {
    GLsizei range = in->Get<GLsizei>();
    GLuint result = glGenLists(range);
    Buffer *out = new Buffer();
    out->Add(result);
    return new Result(0, out);
}

GL_ROUTINE_HANDLER(NewList) {
    GLuint list = in->Get<GLuint>();
    GLenum mode = in->Get<GLenum>();
    glNewList(list, mode);
    return new Result(0);
}

GL_ROUTINE_HANDLER(ShadeModel) {
    GLenum mode = in->Get<GLenum>();
    glShadeModel(mode);
    return new Result(0);
}

GL_ROUTINE_HANDLER(Normal3f) {
    GLfloat nx = in->Get<GLfloat>();
    GLfloat ny = in->Get<GLfloat>();
    GLfloat nz = in->Get<GLfloat>();
    glNormal3f(nx, ny, nz);
    return new Result(0);
}

GL_ROUTINE_HANDLER(Begin) {
    GLenum mode = in->Get<GLenum>();
    glBegin(mode);
    return new Result(0);
}

GL_ROUTINE_HANDLER(Enable) {
    GLenum cap = in->Get<GLenum>();
    glEnable(cap);
    return new Result(0);
}

GL_ROUTINE_HANDLER(Vertex3f) {
    GLfloat x = in->Get<GLfloat>();
    GLfloat y = in->Get<GLfloat>();
    GLfloat z = in->Get<GLfloat>();
    glVertex3f(x, y, z);
    return new Result(0);
}

GL_ROUTINE_HANDLER(End) {
    glEnd();
    return new Result(0);
}

GL_ROUTINE_HANDLER(EndList) {
    glEndList();
    return new Result(0);
}

GL_ROUTINE_HANDLER(Viewport) {
    GLint x = in->Get<GLint>();
    GLint y = in->Get<GLint>();
    GLsizei width = in->Get<GLsizei>();
    GLsizei height = in->Get<GLsizei>();
    glViewport(x, y, width, height);
    return new Result(0);
}

GL_ROUTINE_HANDLER(MatrixMode) {
    GLenum mode = in->Get<GLenum>();
    glMatrixMode(mode);
    return new Result(0);
}

GL_ROUTINE_HANDLER(LoadIdentity) {
    glLoadIdentity();
    return new Result(0);
}

GL_ROUTINE_HANDLER(Frustum) {
    GLdouble left = in->Get<GLdouble>();
    GLdouble right = in->Get<GLdouble>();
    GLdouble bottom = in->Get<GLdouble>();
    GLdouble top = in->Get<GLdouble>();
    GLdouble nearVal = in->Get<GLdouble>();
    GLdouble farVal = in->Get<GLdouble>();
    glFrustum(left, right, bottom, top, nearVal, farVal);
    return new Result(0);
}

GL_ROUTINE_HANDLER(Translatef) {
    GLfloat x = in->Get<GLfloat>();
    GLfloat y = in->Get<GLfloat>();
    GLfloat z = in->Get<GLfloat>();
    glTranslatef(x, y, z);
    return new Result(0);
}

GL_ROUTINE_HANDLER(CallList) {
    GLuint list = in->Get<GLuint>();
    glCallList(list);
    return new Result(0);
}

GL_ROUTINE_HANDLER(Clear) {
    GLbitfield mask = in->Get<GLbitfield>();
    glClear(mask);
    return new Result(0);
}

GL_ROUTINE_HANDLER(PopMatrix) {
    glPopMatrix();
    return new Result(0);
}

GL_ROUTINE_HANDLER(PushMatrix) {
    glPushMatrix();
    return new Result(0);
}

GL_ROUTINE_HANDLER(Rotatef) {
    GLfloat angle = in->Get<GLfloat>();
    GLfloat x = in->Get<GLfloat>();
    GLfloat y = in->Get<GLfloat>();
    GLfloat z = in->Get<GLfloat>();
    glRotatef(angle, x, y, z);
    return new Result(0);
}


int *row = new int[300];
GL_ROUTINE_HANDLER(XSwapBuffers) {
    glXSwapBuffers(dpy, drawable);
    pThis->Lock();
    char *buffer = pThis->GetFramebuffer();
    memset(buffer, 0, 300 * 300 * sizeof(int));
    glReadPixels(0, 0, 300, 300, GL_BGRA, GL_UNSIGNED_BYTE, buffer);
#if 0
    for(int i = 0; i < 300 / 2; i++) {
        memmove(row, buffer + (299 - i) * 300 * sizeof(int), 300 * sizeof(int));
        memmove(buffer + (299 - i) * 300 * sizeof(int), buffer + i * 300 * sizeof(int), 300 * sizeof(int));
        memmove(buffer + i * 300 * sizeof(int), row, 300 * sizeof(int));
    }
#endif
    pThis->Unlock();
    return new Result(0);
}

GL_ROUTINE_HANDLER(Lightfv) {
    GLenum light = in->Get<GLenum>();
    GLenum pname = in->Get<GLenum>();
    GLfloat *params = in->AssignAll<GLfloat>();
    glLightfv(light, pname, params);
    return new Result(0);
}

GL_ROUTINE_HANDLER(Materialfv) {
    GLenum face = in->Get<GLenum>();
    GLenum pname = in->Get<GLenum>();
    GLfloat *params = in->AssignAll<GLfloat>();
    glMaterialfv(face, pname, params);
    return new Result(0);
}

GL_ROUTINE_HANDLER(__ExecuteRoutines) {
    string routine;
    while(!in->Empty()) {
        routine = string(in->AssignString());
        pThis->Execute(routine, in);
    }
    return new Result(0);
}

GL_ROUTINE_HANDLER(__GetBuffer) {
    return new Result(0, new Buffer(pThis->GetFramebuffer(), 300 * 300 * sizeof(int)));
}

GL_ROUTINE_HANDLER(XQueryExtension) {
    int errorBase;
    int eventBase;
    glXQueryExtension(dpy, &errorBase, &eventBase);
    Buffer *out = new Buffer();
    out->Add(errorBase);
    out->Add(eventBase);
    return new Result(0, out);
}
