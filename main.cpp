#include <windows.h>
#include <iostream>
#include <vector>

#include <QtGlobal>
#include <QGuiApplication>
#include <QOpenGLWindow>
#include <QOpenGLFunctions_3_3_Compatibility>
#include <QTimer>
#include <QElapsedTimer>
#include <QDebug>

#include <boost/compute.hpp>
#include <boost/compute/interop/opengl.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION


namespace bc = boost::compute;
namespace br = boost::random;

struct Particle {
    float x, y, z, u, v, w, px[3], py[3], pz[3], gx[3], gy[3], gz[3];
    int   cx, cy, cz;
};

struct Node {
    float m, d, gx, gy, gz, u, v, w, ax, ay, az;
};

uint idx(uint nsize, uint x, uint y, uint z) {
    return nsize*nsize*x + nsize*y + z;
}

void phase1(std::vector<Node> &node) {
    for(uint gid = 0; gid < node.size(); gid++) {
        struct Node* n = &node[gid];
        n->m = 0.0;
        n->d = 0.0;
        n->gx = 0.0;
        n->gy = 0.0;
        n->gz = 0.0;
        n->u = 0.0;
        n->v = 0.0;
        n->w = 0.0;
        n->ax = 0.0;
        n->ay = 0.0;
        n->az = 0.0;
    }
}

void phase2(std::vector<Particle> &particle, std::vector<Node> &node, uint nsize) {
    for(uint gid = 0; gid < particle.size(); gid++) {
        struct Particle* p = &particle[gid];
        p->cx = (int) (p->x - 0.5f);
        p->cy = (int) (p->y - 0.5f);
        p->cz = (int) (p->z - 0.5f);

        float x = p->cx - p->x;
        p->px[0] = 0.5f * x * x + 1.5f * x + 1.125f;
        p->gx[0] = x + 1.5f;
        ++x;
        p->px[1] = -x * x + 0.75f;
        p->gx[1] = -2.0f * x;
        ++x;
        p->px[2] = 0.5f * x * x - 1.5f * x + 1.125f;
        p->gx[2] = x - 1.5f;

        float y = p->cy - p->y;
        p->py[0] = 0.5f * y * y + 1.5f * y + 1.125f;
        p->gy[0] = y + 1.5f;
        ++y;
        p->py[1] = -y * y + 0.75f;
        p->gy[1] = -2.0f * y;
        ++y;
        p->py[2] = 0.5f * y * y - 1.5f * y + 1.125f;
        p->gy[2] = y - 1.5f;

        float z = p->cz - p->z;
        p->pz[0] = 0.5f * z * z + 1.5f * z + 1.125f;
        p->gz[0] = z + 1.5f;
        ++z;
        p->pz[1] = -z * z + 0.75f;
        p->gz[1] = -2.0f * z;
        ++z;
        p->pz[2] = 0.5f * z * z - 1.5f * z + 1.125f;
        p->gz[2] = z - 1.5f;

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    int cxi = p->cx + i;
                    int cyj = p->cy + j;
                    int czk = p->cz + k;
                    struct Node* n = &node[idx(nsize, cxi, cyj, czk)];
                    float phi = p->px[i] * p->py[j] * p->pz[k];
                    n->m += phi;
                    n->d += phi;
                    float dx = p->gx[i] * p->py[j] * p->pz[k];
                    float dy = p->px[i] * p->gy[j] * p->pz[k];
                    float dz = p->px[i] * p->py[j] * p->gz[k];
                    n->gx += dx;
                    n->gy += dy;
                    n->gz += dz;
                }
            }
        }
    }
}

void phase3(std::vector<Particle> &particle, std::vector<Node> &node, uint nsize) {
    for(uint gid = 0; gid < particle.size(); gid++) {
        struct Particle* p = &particle[gid];
        int cx = (int) p->x;
        int cy = (int) p->y;
        int cz = (int) p->z;
        int cxi = cx + 1;
        int cyi = cy + 1;
        int czi = cz + 1;
        float u = p->x - cx;
        float v = p->y - cy;
        float w = p->z - cz;

        float p000 = node[idx(nsize,cx,cy,cz)].d;
        float x000 = node[idx(nsize,cx,cy,cz)].gx;
        float y000 = node[idx(nsize,cx,cy,cz)].gy;
        float z000 = node[idx(nsize,cx,cy,cz)].gz;
        float p001 = node[idx(nsize,cx,cy,czi)].d;
        float x001 = node[idx(nsize,cx,cy,czi)].gx;
        float y001 = node[idx(nsize,cx,cy,czi)].gy;
        float z001 = node[idx(nsize,cx,cy,czi)].gz;

        float p010 = node[idx(nsize,cx,cyi,cz)].d;
        float x010 = node[idx(nsize,cx,cyi,cz)].gx;
        float y010 = node[idx(nsize,cx,cyi,cz)].gy;
        float z010 = node[idx(nsize,cx,cyi,cz)].gz;
        float p011 = node[idx(nsize,cx,cyi,czi)].d;
        float x011 = node[idx(nsize,cx,cyi,czi)].gx;
        float y011 = node[idx(nsize,cx,cyi,czi)].gy;
        float z011 = node[idx(nsize,cx,cyi,czi)].gz;

        float p100 = node[idx(nsize,cxi,cy,cz)].d;
        float x100 = node[idx(nsize,cxi,cy,cz)].gx;
        float y100 = node[idx(nsize,cxi,cy,cz)].gy;
        float z100 = node[idx(nsize,cxi,cy,cz)].gz;
        float p101 = node[idx(nsize,cxi,cy,czi)].d;
        float x101 = node[idx(nsize,cxi,cy,czi)].gx;
        float y101 = node[idx(nsize,cxi,cy,czi)].gy;
        float z101 = node[idx(nsize,cxi,cy,czi)].gz;

        float p110 = node[idx(nsize,cxi,cyi,cz)].d;
        float x110 = node[idx(nsize,cxi,cyi,cz)].gx;
        float y110 = node[idx(nsize,cxi,cyi,cz)].gy;
        float z110 = node[idx(nsize,cxi,cyi,cz)].gz;
        float p111 = node[idx(nsize,cxi,cyi,czi)].d;
        float x111 = node[idx(nsize,cxi,cyi,czi)].gx;
        float y111 = node[idx(nsize,cxi,cyi,czi)].gy;
        float z111 = node[idx(nsize,cxi,cyi,czi)].gz;

        float dx00 = p100 - p000;
        float dx01 = p101 - p001;
        float dx10 = p110 - p010;

        float dy00 = p010 - p000;
        float dy01 = p011 - p001;
        float dy10 = p110 - p100;

        float dz00 = p001 - p000;
        float dz01 = p011 - p010;
        float dz10 = p101 - p100;

        float C000 = p000;
        float C100 = x000;
        float C010 = y000;
        float C001 = z000;

        float C310 = x110 - x100 + x010 - x000 - 2.0f * (dx10 - dx00);
        float C210 = 3.0f * (dx10 - dx00) - 2.0f * (x010 - x000) - (x110 - x100);

        float C301 = x101 - x100 + x001 - x000 - 2.0f * (dx01 - dx00);
        float C201 = 3.0f * (dx01 - dx00) - 2.0f * (x001 - x000) - (x101 - x100);

        float C130 = y110 - y010 + y100 - y000 - 2.0f * (dy10 - dy00);
        float C120 = 3.0f * (dy10 - dy00) - 2.0f * (y100 - y000) - (y110 - y010);

        float C031 = y011 - y010 + y001 - y000 - 2.0f * (dy01 - dy00);
        float C021 = 3.0f * (dy01 - dy00) - 2.0f * (y001 - y000) - (y011 - y010);

        float C103 = z101 - z001 + z100 - z000 - 2.0f * (dz10 - dz00);
        float C102 = 3.0f * (dz10 - dz00) - 2.0f * (z100 - z000) - (z101 - z001);

        float C013 = z011 - z001 + z010 - z000 - 2.0f * (dz01 - dz00);
        float C012 = 3.0f * (dz01 - dz00) - 2.0f * (z010 - z000) - (z011 - z001);

        float C300 = x100 + x000 - 2.0f * dx00;
        float C200 = 3.0f * dx00 - x100 - 2.0f * x000;

        float C030 = y010 + y000 - 2.0f * dy00;
        float C020 = 3.0f * dy00 - y010 - 2.0f * y000;

        float C003 = z001 + z000 - 2.0f * dz00;
        float C002 = 3.0f * dz00 - z001 - 2.0f * z000;

        float C110 = x010 - C100 - C120 - C130;
        float C011 = y001 - C010 - C012 - C013;
        float C101 = z100 - C001 - C201 - C301;

        float A = p100 + y100 + z100 + C011 + C020 + C002 + C120 + C021 + C102 + C012 + C030 + C003 + C130 + C031 + C103 + C013;

        float f111_A = p111 - A;

        float x0 = x111 - x110 - x101 + x100;
        float x1 = x011 - x010 - x001 + x000;
        float C311 = x0 + x1 - 2.0f * f111_A;
        float C211 = 3.0f * f111_A - x0 - 2.0f * x1;

        float y0 = y111 - y110 - y011 + y010;
        float y1 = y101 - y100 - y001 + y000;
        float C131 = y0 + y1 - 2.0f * f111_A;
        float C121 = 3.0f * f111_A - y0 - 2.0f * y1;

        float z0 = z111 - z101 - z011 + z001;
        float z1 = z110 - z100 - z010 + z000;
        float C113 = z0 + z1 - 2.0f * f111_A;
        float C112 = 3.0f * f111_A - z0 - 2.0f * z1;

        float C111 = x1 + y1 + z1 - 2.0f * f111_A;

        float density = C000 + (C001 + (C002 + C003 * w) * w) * w + (C010 + (C011 + (C012 + C013 * w) * w) * w) * v
                + (C020 + C021 * w) * v * v + (C030 + C031 * w) * v * v * v
                + (C100 + (C110 + (C120 + C130 * v) * v) * v + (C101 + (C111 + (C121 + C131 * v) * v) * v) * w
                   + (C102 + C112 * v) * w * w + (C103 + C113 * v) * w * w * w) * u
                + (C200 + C210 * v + (C201 + C211 * v) * w) * u * u
                + (C300 + C310 * v + (C301 + C311 * v) * w) * u * u * u;

        float pressure = 1.0 * (density - 1.0);
        float fx = 0.0f;
        float fy = 0.0f;
        float fz = 0.0f;

        // Rebound or reflect after striking a surface
        if (p->x < 3.0f) {
            fx += (3.0f - p->x);
        } else if (p->x > nsize - 4) {
            fx += (nsize - 4 - p->x);
        }
        if (p->y < 3.0f) {
            fy += (3.0f - p->y);
        } else if (p->y > nsize - 4) {
            fy += (nsize - 4 - p->y);
        }
        if (p->z < 3.0f) {
            fz += (3.0f - p->z);
        } else if (p->z > nsize - 4) {
            fz += (nsize - 4 - p->z);
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    struct Node* n = &node[idx(nsize, p->cx + i, p->cy + j, p->cz + k)];
                    float phi = p->px[i] * p->py[j] * p->pz[k];
                    float gx = p->gx[i] * p->py[j] * p->pz[k];
                    float gy = p->px[i] * p->gy[j] * p->pz[k];
                    float gz = p->px[i] * p->py[j] * p->gz[k];
                    // Total force
                    n->ax += -(gx * pressure) + fx * phi;
                    n->ay += -(gy * pressure) + fy * phi;
                    n->az += -(gz * pressure) + fz * phi;
                }
            }
        }
    }
}

void phase4(std::vector<Node> &node) {
    for(uint gid = 0; gid < node.size(); gid++) {
        struct Node* n = &node[gid];
        if (n->m > 0.0f) {
            n->ax /= n->m;
            n->ay /= n->m;
            n->az /= n->m;
            n->ay -= 0.1f; // gravity
        }
    }
}

void phase5(std::vector<Particle> &particle, std::vector<Node> &node, uint nsize) {
    for(uint gid = 0; gid < particle.size(); gid++) {
        struct Particle* p = &particle[gid];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    struct Node* n = &node[idx(nsize, p->cx + i, p->cy + j, p->cz + k)];
                    float phi = p->px[i] * p->py[j] * p->pz[k];
                    // Particle velocity
                    p->u += phi * n->ax;
                    p->v += phi * n->ay;
                    p->w += phi * n->az;
                }
            }
        }
        // Particle momentum
        float mu = p->u;
        float mv = p->v;
        float mw = p->w;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    struct Node* n = &node[idx(nsize, p->cx + i, p->cy + j, p->cz + k)];
                    float phi = p->px[i] * p->py[j] * p->pz[k];
                    // Node velocity
                    n->u += phi * mu;
                    n->v += phi * mv;
                    n->w += phi * mw;
                }
            }
        }
    }
}

void phase6(std::vector<Node> &node) {
    for(uint gid = 0; gid < node.size(); gid++) {
        struct Node* n = &node[gid];
        if (n->m > 0.0f) {
            n->u /= n->m;
            n->v /= n->m;
            n->w /= n->m;
        }
    }
}

void phase7(std::vector<Particle> &particle, std::vector<Node> &node, uint nsize) {
    for(uint gid = 0; gid < particle.size(); gid++) {
        struct Particle* p = &particle[gid];
        float gu = 0.0f;
        float gv = 0.0f;
        float gw = 0.0f;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    struct Node* n = &node[idx(nsize, p->cx + i, p->cy + j, p->cz + k)];
                    float phi = p->px[i] * p->py[j] * p->pz[k];
                    // Particle velocity
                    gu += phi * n->u;
                    gv += phi * n->v;
                    gw += phi * n->w;
                }
            }
        }
        // Particle Position
        p->x += gu;
        p->y += gv;
        p->z += gw;
        p->u += 1.0f * (gu - p->u);
        p->v += 1.0f * (gv - p->v);
        p->w += 1.0f * (gw - p->w);

        // If particle move outsite box, set to border
        if (p->x < 1.0f) {
            p->x = 1.0f;
            p->u = 0.0f;
        } else if (p->x > nsize - 2) {
            p->x = nsize - 2;
            p->u = 0.0f;
        }
        if (p->y < 1.0f) {
            p->y = 1.0f;
            p->v = 0.0f;
        } else if (p->y > nsize - 2) {
            p->y = nsize - 2;
            p->v = 0.0f;
        }
        if (p->z < 1.0f) {
            p->z = 1.0f;
            p->w = 0.0f;
        } else if (p->z > nsize - 2) {
            p->z = nsize - 2;
            p->w = 0.0f;
        }
    }
}

void transform1(std::vector<Particle> &particle, uint nsize) {
    for(uint gid = 0; gid < particle.size(); gid++) {
        struct Particle* p = &particle[gid];
        float scale = (nsize/2.0);
        p->x = (p->x + 1.0)*scale;
        p->y = (p->y + 1.0)*scale;
        p->z = (p->z + 1.0)*scale;
    }
}

void transform2(std::vector<Particle> &particle, uint nsize) {
    for(uint gid = 0; gid < particle.size(); gid++) {
        struct Particle* p = &particle[gid];
        float scale = (nsize/2.0);
        p->x = (p->x)/scale - 1.0;
        p->y = (p->y)/scale - 1.0;
        p->z = (p->z)/scale - 1.0;
    }
}



class FluidWindow : public QOpenGLWindow,
                    protected QOpenGLFunctions_3_3_Compatibility
{
    Q_OBJECT

public:
    FluidWindow(size_t particles, size_t nsize);
    ~FluidWindow();

    void initializeGL();
    void resizeGL(int width, int height);
    void paintGL();
    void updateParticles();

private:
    QTimer* runner;
    QElapsedTimer* timer;
    int frame;
    std::vector<Particle>* m_particle;
    std::vector<Node>* m_node;

    bool m_initial_draw;
    GLuint vbo;

    size_t num_particle, nsize;
};

FluidWindow::FluidWindow(size_t particles, size_t nsize)
    : m_initial_draw(true), num_particle(particles), nsize(nsize)
{
    // create a runner to redraw as soon as possible
    runner = new QTimer(this);
    connect(runner, SIGNAL(timeout()), this, SLOT(update()));
    runner->start();

    // create a timer to calculate elapsed time for a frame
    timer = new QElapsedTimer();
    timer->start();
    frame = 0;
}

FluidWindow::~FluidWindow()
{
    delete m_particle;
    delete m_node;
}

void FluidWindow::initializeGL()
{
    initializeOpenGLFunctions();

    // prepare random particle positions that will be transferred to the vbo
    Particle* temp = new Particle[num_particle];
    br::uniform_real_distribution<float> dist(-0.3f, 0.2f);
    br::mt19937_64 gen;
    for(size_t i = 0; i < num_particle; i++) {
        temp[i].x = dist(gen);
        temp[i].y = dist(gen);
        temp[i].z = dist(gen);
    }
//    for (uint i = 0; i < 20; ++i) {
//        for (uint j = 0; j < 25; ++j) {
//            for (uint k = 0; k < 20; ++k) {
//                uint idx = i*25*20 + j*20 + k;
//                float scale = nsize/2.0f;
//                temp[idx].x = (i+4)/scale - 1.0f;
//                temp[idx].y = (j+4)/scale - 1.0f;
//                temp[idx].z = (k+4)/scale - 1.0f;
//                //std::cout << idx << " : " << temp[idx].x << " " << temp[idx].y << " " << temp[idx].z << std::endl;
//            }
//        }
//    }

    m_particle = new std::vector<Particle>(temp, temp+num_particle);

    // prepare node
    m_node = new std::vector<Node>(nsize*nsize*nsize);

    vbo = 0;
    glGenBuffers(1, &vbo);
}

void FluidWindow::resizeGL(int width, int height)
{
    // update viewport
    glViewport(0, 0, width, height);
}

void FluidWindow::paintGL()
{
    // clear buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // check if this is the first draw
    if(m_initial_draw) {
        // do not update particles
        m_initial_draw = false;
    } else {
        // update particles
        updateParticles();
    }

    // create an OpenGL vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, num_particle*sizeof(Particle), &m_particle->front(), GL_DYNAMIC_DRAW);

    // draw
    glVertexPointer(3, GL_FLOAT, 108, nullptr);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, num_particle);
    glFinish();

    frame++;
    qint64 time = timer->elapsed();
    if (time >= 1000) {
        qDebug("FPS: %.2f", (float)frame / (time/1000.0));
        frame = 0;
        timer->start();
    }
}

void FluidWindow::updateParticles()
{
    transform1(*m_particle, nsize);
    phase1(*m_node);
    phase2(*m_particle, *m_node, nsize);
    phase3(*m_particle, *m_node, nsize);
    phase4(*m_node);
    phase5(*m_particle, *m_node, nsize);
    phase6(*m_node);
    phase7(*m_particle, *m_node, nsize);
    transform2(*m_particle, nsize);
}

int main(int argc, char *argv[])
{
    size_t particles = 10000, glength = 40;
    QGuiApplication app(argc, argv);
    FluidWindow fluid(particles, glength);
    fluid.setWidth(600);
    fluid.setHeight(600);
    fluid.show();
    return app.exec();
}

#include "main.moc"
