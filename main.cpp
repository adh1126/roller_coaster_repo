// roller_coaster.cpp
// Simple 3D Roller Coaster Ride Simulation using OpenGL + GLUT (legacy)
// Features: Catmull-Rom track, carriage, first/third person camera, speed control.
// Compile: g++ roller_coaster.cpp -o roller_coaster -lGL -lglut -std=c++17

#include <GL/glut.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

struct Vec3 {
    float x, y, z;
    Vec3(): x(0), y(0), z(0) {}
    Vec3(float X,float Y,float Z): x(X), y(Y), z(Z) {}
    Vec3 operator+(const Vec3& o) const { return Vec3(x+o.x,y+o.y,z+o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x-o.x,y-o.y,z-o.z); }
    Vec3 operator*(float s) const { return Vec3(x*s,y*s,z*s); }
    Vec3 operator/(float s) const { return Vec3(x/s,y/s,z/s); }
};

inline float dot(const Vec3 &a, const Vec3 &b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3 cross(const Vec3 &a, const Vec3 &b){ return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }
inline float length(const Vec3 &v){ return std::sqrt(dot(v,v)); }
inline Vec3 normalize(const Vec3 &v){ float L = length(v); if (L==0) return v; return v / L; }

// ---------- Simulation / track parameters ----------
int windowW = 1024, windowH = 700;

std::vector<Vec3> controlPoints; // control points for Catmull-Rom (closed)
int samplesPerSegment = 80; // how many samples to compute per segment
std::vector<Vec3> trackPoints; // sampled points along spline
std::vector<Vec3> trackTangents; // tangent at sample points
std::vector<Vec3> trackNormals; // normal (for banking or track cross-section)

float tGlobal = 0.0f; // progress along track in [0, 1)
float speed = 0.08f; // units: fraction per second (adjusted by dt)
bool paused = false;
bool firstPerson = false; // camera mode
float carriageHeight = 0.3f; // offset above track center
float trackRadius = 0.07f; // radius of track tube for rendering
float timePrev = 0.0f;

// ---------- Utility: Catmull-Rom interpolation ----------
Vec3 catmullRom(const Vec3 &p0, const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, float t) {
    // tension = 0.5 standard Catmull-Rom
    float t2 = t * t;
    float t3 = t2 * t;
    Vec3 res;
    res.x = 0.5f * ((2.0f*p1.x) +
                    (-p0.x + p2.x)*t +
                    (2.0f*p0.x - 5.0f*p1.x + 4.0f*p2.x - p3.x)*t2 +
                    (-p0.x + 3.0f*p1.x - 3.0f*p2.x + p3.x)*t3);
    res.y = 0.5f * ((2.0f*p1.y) +
                    (-p0.y + p2.y)*t +
                    (2.0f*p0.y - 5.0f*p1.y + 4.0f*p2.y - p3.y)*t2 +
                    (-p0.y + 3.0f*p1.y - 3.0f*p2.y + p3.y)*t3);
    res.z = 0.5f * ((2.0f*p1.z) +
                    (-p0.z + p2.z)*t +
                    (2.0f*p0.z - 5.0f*p1.z + 4.0f*p2.z - p3.z)*t2 +
                    (-p0.z + 3.0f*p1.z - 3.0f*p2.z + p3.z)*t3);
    return res;
}

// derivative of Catmull-Rom (for tangent)
Vec3 catmullRomTangent(const Vec3 &p0,const Vec3 &p1,const Vec3 &p2,const Vec3 &p3,float t) {
    float t2 = t * t;
    Vec3 res;
    res.x = 0.5f * ((-p0.x + p2.x) + 2*(2.0f*p0.x - 5.0f*p1.x + 4.0f*p2.x - p3.x)*t + 3*(-p0.x + 3.0f*p1.x - 3.0f*p2.x + p3.x)*t2);
    res.y = 0.5f * ((-p0.y + p2.y) + 2*(2.0f*p0.y - 5.0f*p1.y + 4.0f*p2.y - p3.y)*t + 3*(-p0.y + 3.0f*p1.y - 3.0f*p2.y + p3.y)*t2);
    res.z = 0.5f * ((-p0.z + p2.z) + 2*(2.0f*p0.z - 5.0f*p1.z + 4.0f*p2.z - p3.z)*t + 3*(-p0.z + 3.0f*p1.z - 3.0f*p2.z + p3.z)*t2);
    return res;
}

// build sampled track (closed loop)
void buildTrack() {
    trackPoints.clear();
    trackTangents.clear();
    trackNormals.clear();
    int n = (int)controlPoints.size();
    if (n < 4) return;

    for (int i = 0; i < n; ++i) {
        // segment from p1 to p2 where indices are (i-1,i,i+1,i+2)
        Vec3 p0 = controlPoints[(i - 1 + n) % n];
        Vec3 p1 = controlPoints[i];
        Vec3 p2 = controlPoints[(i + 1) % n];
        Vec3 p3 = controlPoints[(i + 2) % n];

        for (int s = 0; s < samplesPerSegment; ++s) {
            float t = float(s) / float(samplesPerSegment);
            Vec3 pos = catmullRom(p0,p1,p2,p3,t);
            Vec3 tan = catmullRomTangent(p0,p1,p2,p3,t);
            tan = normalize(tan);
            trackPoints.push_back(pos);
            trackTangents.push_back(tan);
        }
    }
    // compute normals using Frenet frame-ish method:
    // pick an initial up vector and compute successive normals by Gram-Schmidt
    Vec3 up(0,1,0);
    for (size_t i=0;i<trackPoints.size();++i){
        Vec3 T = trackTangents[i];
        Vec3 N = cross(T, cross(up, T)); // project up into normal plane
        if (length(N) < 1e-4f) {
            // fallback: choose different up
            N = cross(T, Vec3(1,0,0));
            if (length(N) < 1e-4f) N = cross(T, Vec3(0,0,1));
        }
        N = normalize(N);
        trackNormals.push_back(N);
    }
}

// get position and tangent for param u in [0,1)
void sampleTrack(float u, Vec3 &pos, Vec3 &tan, Vec3 &normal) {
    if (trackPoints.empty()) { pos = Vec3(); tan = Vec3(0,0,1); normal = Vec3(0,1,0); return; }
    float total = (float)trackPoints.size();
    float idx = u * total;
    if (idx < 0) idx += total;
    int i0 = (int)floor(idx) % (int)trackPoints.size();
    int i1 = (i0 + 1) % (int)trackPoints.size();
    float localT = idx - floor(idx);
    // simple lerp between neighboring samples (could do smoother interpolation)
    Vec3 p0 = trackPoints[i0], p1 = trackPoints[i1];
    pos = p0*(1.0f-localT) + p1*(localT);
    Vec3 t0 = trackTangents[i0], t1 = trackTangents[i1];
    tan = normalize(t0*(1.0f-localT) + t1*(localT));
    Vec3 n0 = trackNormals[i0], n1 = trackNormals[i1];
    normal = normalize(n0*(1.0f-localT) + n1*(localT));
}

// ---------- simple lookAt implementation (no GLU) ----------
// builds a modelview matrix that matches gluLookAt and loads it
void myLookAt(const Vec3 &eye, const Vec3 &center, const Vec3 &up) {
    Vec3 F = normalize(center - eye);
    Vec3 S = normalize(cross(F, up));
    Vec3 U = cross(S, F);

    // Column-major (OpenGL expects column-major when loading with glMultMatrixf)
    float M[16] = {
        S.x, U.x, -F.x, 0.0f,
        S.y, U.y, -F.y, 0.0f,
        S.z, U.z, -F.z, 0.0f,
        0.0f,0.0f,0.0f,1.0f
    };

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMultMatrixf(M);
    glTranslatef(-eye.x, -eye.y, -eye.z);
}

// ---------- Rendering helpers ----------
void drawGround() {
    const float size = 100.0f;
    const float step = 2.0f;
    glDisable(GL_LIGHTING);
    glColor3f(0.25f,0.8f,0.25f);
    glBegin(GL_QUADS);
        glVertex3f(-size, -0.1f, -size);
        glVertex3f(size, -0.1f, -size);
        glVertex3f(size, -0.1f, size);
        glVertex3f(-size, -0.1f, size);
    glEnd();
    // grid lines
    glColor3f(0.2f,0.6f,0.2f);
    glBegin(GL_LINES);
    for (float x=-size; x<=size; x+=step) {
        glVertex3f(x, -0.09f, -size);
        glVertex3f(x, -0.09f, size);
    }
    for (float z=-size; z<=size; z+=step) {
        glVertex3f(-size, -0.09f, z);
        glVertex3f(size, -0.09f, z);
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

void drawTrack() {
    // draw a simple ribbon/tube by extruding a circle cross-section using track normals/tangents
    int segsAround = 8;
    for (size_t i=0;i<trackPoints.size();++i){
        Vec3 p = trackPoints[i];
        Vec3 T = trackTangents[i];
        Vec3 N = trackNormals[i];
        Vec3 B = normalize(cross(T,N)); // binormal
        // draw ring as triangle strip between this and next sample
        size_t j = (i+1) % trackPoints.size();
        Vec3 p2 = trackPoints[j];
        Vec3 T2 = trackTangents[j];
        Vec3 N2 = trackNormals[j];
        Vec3 B2 = normalize(cross(T2,N2));
        glBegin(GL_QUAD_STRIP);
        for (int k=0;k<=segsAround;k++){
            float angle = (float)k / segsAround * 2.0f * M_PI;
            float ca = cos(angle), sa = sin(angle);
            Vec3 offset = N* (ca*trackRadius) + B*(sa*trackRadius);
            Vec3 offset2 = N2* (ca*trackRadius) + B2*(sa*trackRadius);
            Vec3 v1 = p + offset;
            Vec3 v2 = p2 + offset2;
            // normal for lighting approximate as offset normalized
            Vec3 n1 = normalize(offset);
            glNormal3f(n1.x, n1.y, n1.z);
            glVertex3f(v1.x, v1.y, v1.z);
            Vec3 n2 = normalize(offset2);
            glNormal3f(n2.x, n2.y, n2.z);
            glVertex3f(v2.x, v2.y, v2.z);
        }
        glEnd();
    }

    // rails: draw two rails offset on either side of the track center along normal
    glDisable(GL_LIGHTING);
    glLineWidth(4.0f);
    glColor3f(0.2f,0.2f,0.2f);
    glBegin(GL_LINE_STRIP);
    for (size_t i=0;i<trackPoints.size();++i){
        Vec3 p = trackPoints[i];
        Vec3 N = trackNormals[i];
        Vec3 offset = N * (0.15f);
        glVertex3f(p.x + offset.x, p.y + offset.y, p.z + offset.z);
    }
    glEnd();

    glBegin(GL_LINE_STRIP);
    for (size_t i=0;i<trackPoints.size();++i){
        Vec3 p = trackPoints[i];
        Vec3 N = trackNormals[i];
        Vec3 offset = N * (-0.15f);
        glVertex3f(p.x + offset.x, p.y + offset.y, p.z + offset.z);
    }
    glEnd();
    glLineWidth(1.0f);
    glEnable(GL_LIGHTING);
}

void drawCarriage(const Vec3 &pos, const Vec3 &tan, const Vec3 &normal) {
    // build local frame: T (forward), N (side), B (up)
    Vec3 T = normalize(tan);
    Vec3 N = normalize(normal);
    Vec3 B = normalize(cross(T, N));
    // Slightly adjust N,B to be orthonormal
    N = normalize(cross(B, T));
    B = normalize(cross(T, N));

    // Create transformation matrix (3x3 rotation + translation)
    // Column-major for glMultMatrixf
    float M[16] = {
        N.x, N.y, N.z, 0.0f,
        B.x, B.y, B.z, 0.0f,
        T.x, T.y, T.z, 0.0f,
        pos.x, pos.y, pos.z, 1.0f
    };

    glPushMatrix();
    glMultMatrixf(M);

    // raise a bit so wheels under carriage
    glTranslatef(0.0f, 0.2f, 0.0f);

    // carriage body
    glColor3f(0.9f, 0.2f, 0.2f);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (GLfloat[]){0.9f,0.2f,0.2f,1.0f});
    glutSolidCube(0.6f);

    // wheels (simple tori or cylinders)
    glColor3f(0.1f,0.1f,0.1f);
    glPushMatrix();
      glTranslatef(0.28f, -0.25f, 0.18f);
      glutSolidSphere(0.12,12,12);
    glPopMatrix();
    glPushMatrix();
      glTranslatef(-0.28f, -0.25f, 0.18f);
      glutSolidSphere(0.12,12,12);
    glPopMatrix();
    glPushMatrix();
      glTranslatef(0.28f, -0.25f, -0.18f);
      glutSolidSphere(0.12,12,12);
    glPopMatrix();
    glPushMatrix();
      glTranslatef(-0.28f, -0.25f, -0.18f);
      glutSolidSphere(0.12,12,12);
    glPopMatrix();

    glPopMatrix();
}

// ---------- Input / Controls ----------
void resetSimulation() {
    tGlobal = 0.0f;
    speed = 0.08f;
    paused = false;
    firstPerson = false;
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27: exit(0); break;
        case ' ': paused = !paused; break;
        case '+': speed *= 1.2f; break;
        case '-': speed *= 0.8f; break;
        case 'f': case 'F': firstPerson = !firstPerson; break;
        case 'r': case 'R': resetSimulation(); break;
    }
}

// ---------- Main display & update ----------
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // camera / view
    Vec3 pos, tan, normal;
    sampleTrack(tGlobal, pos, tan, normal);

    if (firstPerson) {
        // position camera slightly above the track center and a bit behind the carriage center
        Vec3 camPos = pos + normal * carriageHeight - tan * 0.5f;
        Vec3 lookAt = pos + tan * 3.0f; // look forward along the track
        myLookAt(camPos, lookAt, normal);
    } else {
        // third-person: place camera behind and above the carriage, looking at it
        Vec3 camPos = pos + normal * 3.5f - tan * 5.0f + Vec3(0.0f, 1.2f, 0.0f);
        Vec3 lookAt = pos + normal * 0.2f;
        myLookAt(camPos, lookAt, Vec3(0,1,0));
    }

    // lighting
    GLfloat light_pos[] = {10.0f, 10.0f, 10.0f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);

    // scene
    drawGround();
    drawTrack();

    // draw carriage at tGlobal
    Vec3 carPos = pos + normal * carriageHeight;
    drawCarriage(carPos, tan, normal);

    // HUD overlay (use orthographic projection)
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
      glLoadIdentity();
      gluOrtho2D(0, windowW, 0, windowH); // we use gluOrtho2D which is part of GLU; if linking fails, see comment below
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
        glLoadIdentity();
        glDisable(GL_LIGHTING);
        glColor3f(1,1,1);
        char buf[256];
        sprintf(buf, "Mode: %s   Speed: %.3f   t: %.3f   Samples: %zu",
                firstPerson ? "First-Person (F to toggle)" : "Third-Person (F to toggle)",
                speed, tGlobal, trackPoints.size());
        glRasterPos2i(10, windowH - 20);
        for (char *c = buf; *c; ++c) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);
        glEnable(GL_LIGHTING);
      glPopMatrix();
      glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    glutSwapBuffers();
}

void idleFunc() {
    float tNow = (float)glutGet(GLUT_ELAPSED_TIME) * 0.001f; // seconds
    if (timePrev == 0.0f) timePrev = tNow;
    float dt = tNow - timePrev;
    timePrev = tNow;
    if (!paused) {
        // advance tGlobal by speed * dt / trackLengthNormalization
        // track parameter u is [0,1), where 1 = full loop. speed is fraction/sec of loop.
        tGlobal += speed * dt;
        while (tGlobal >= 1.0f) tGlobal -= 1.0f;
        while (tGlobal < 0.0f) tGlobal += 1.0f;
    }
    glutPostRedisplay();
}

// ---------- Window / GL init ----------
void initGL() {
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_NORMALIZE);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    GLfloat amb[] = {0.2f,0.2f,0.2f,1.0f};
    GLfloat diff[] = {0.9f,0.9f,0.9f,1.0f};
    GLfloat spec[] = {0.2f,0.2f,0.2f,1.0f};
    glLightfv(GL_LIGHT0, GL_AMBIENT, amb);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diff);
    glLightfv(GL_LIGHT0, GL_SPECULAR, spec);

    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    glClearColor(0.52f, 0.8f, 0.98f, 1.0f);
}

// reshape
void reshape(int w, int h) {
    windowW = w; windowH = h;
    glViewport(0,0,w,h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspect = (float)w / (float)h;
    gluPerspective(60.0, aspect, 0.1, 200.0); // uses GLU; if you prefer avoid GLU, replace with own perspective matrix
    glMatrixMode(GL_MODELVIEW);
}

// ---------- Setup control points (example loop) ----------
void setupControlPoints() {
    controlPoints.clear();
    // a fun loop around hills and drops
    controlPoints.push_back(Vec3(-8, 1.0f, -10));
    controlPoints.push_back(Vec3(-5, 3.5f, -5));
    controlPoints.push_back(Vec3(-2, 2.0f, -2));
    controlPoints.push_back(Vec3(0, 5.0f, 0));
    controlPoints.push_back(Vec3(3, 2.0f, 2));
    controlPoints.push_back(Vec3(6, 4.0f, 5));
    controlPoints.push_back(Vec3(9, 1.0f, 8));
    controlPoints.push_back(Vec3(6, 0.5f, 10));
    controlPoints.push_back(Vec3(1, 1.2f, 9));
    controlPoints.push_back(Vec3(-3, 2.8f, 6));
    controlPoints.push_back(Vec3(-7, 0.8f, 2));
    // closed loop: duplicate first two points at end is not necessary since we index mod n
}

// ---------- Main ----------
int main(int argc, char** argv) {
    srand((unsigned)time(nullptr));

    // setup control points & track
    setupControlPoints();
    buildTrack();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(windowW, windowH);
    glutCreateWindow("Roller Coaster Ride Simulation");

    initGL();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutIdleFunc(idleFunc);

    printf("Controls:\n");
    printf("  Space : Pause/Resume\n");
    printf("  + / - : Speed up / slow down\n");
    printf("  F     : Toggle First/Third person camera\n");
    printf("  R     : Reset\n");
    printf("  Esc   : Exit\n");

    glutMainLoop();
    return 0;
}
