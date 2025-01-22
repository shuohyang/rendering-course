// This code is based on the lab 2 on carmen and other code
// Part of this code is highly adapted by the mario.js in the given tutorial
// The imported model is from https://rigmodels.com/model.php?view=model__UK1SCZNSF1DXVR7IKGGYOM6UO&ykey=d079ac71ab8bbebe548a462936aaff67&yformat=Json&ysetting1=1&ysetting2=100
var gl;
var shaderProgram;

var canvastopoffset;
var canvasleftoffset;

function initGL(canvas) {
    try {
        gl = canvas.getContext("experimental-webgl");
        gl.viewportWidth=canvas.width;
        gl.viewportHeight=canvas.height;

    } catch(e) {

    }
    if (!gl) {
        alert("Cannot load webGL");
    }
}

//javascript string based on https://www.scaler.com/topics/line-break-in-javascript/
// shader code from simple-triangles.html
var fragshadersource = `
precision highp float;

varying vec4 vColor;
varying vec4 eye_pos;
varying vec4 textureVec;

varying vec4 light_pos;
varying vec4 view_pos;
uniform vec3 camera_pos;
uniform vec4 ambient_mat;
uniform vec4 ambient_light;

uniform vec4 diffuse_mat;
uniform vec4 diffuse_light;

uniform vec4 specular_mat;
uniform vec4 specular_light;

varying vec3 norm_vec;
varying vec3 neg_norm;
varying vec3 w_norm;
uniform float shine_val;

uniform sampler2D tex;
uniform samplerCube cube_texture;

uniform int reflectTexture;

varying vec3 tang_light_pos;
varying vec3 tang_eye_pos;
varying vec3 tang_frag_pos;

uniform sampler2D texdepth;
uniform sampler2D texnormal;

uniform int udrawTexture;

uniform mat4 unMatrix;
uniform mat4 uv2wMatrix; // Based on cubeMappedTeapot.html in the given tutorial

varying vec2 pass_bump;

void main(void) {
    // Shader code is highly adapted by the mario-per-frag.html and mario.html in the given tutorial
    vec3 light_vector = normalize(vec3(light_pos - eye_pos));
    vec4 ambient = ambient_mat * ambient_light;

    float ndotl = max(dot(norm_vec, light_vector), 0.0);
    vec4 diffuse = diffuse_mat * diffuse_light * ndotl;

    vec3 neg_eye_pos = normalize(-vec3(eye_pos));

    vec3 R = normalize(2.0 * ndotl * norm_vec - neg_eye_pos);
    float Rdotval = dot(R, neg_eye_pos);

    float rdotv = max(Rdotval, 0.0);

    vec4 specular;
    if (ndotl > 0.0) {
        specular = specular_mat * specular_light * pow(rdotv, shine_val);
    } else {
        specular = vec4(0, 0, 0, 1);
    }


    int drawTexture = udrawTexture;
    // If statement ? https://stackoverflow.com/questions/20220361/webgl-fragment-shader-not-branching-correctly-on-if-statement

    vec4 col;

    /*
    // Adapte code based on the website https://apoorvaj.io/exploring-bump-mapping-with-webgl/
    // Parallax mapping
    */
    // adapted from the mario-uv-per-frag.html
    // Set color based on Slides
    //gl_FragColor = ambient + diffuse + specular;
    // Draw reflection texture
    //
    // Normal mapping based on website https://apoorvaj.io/exploring-bump-mapping-with-webgl/

    vec4 color;

    if (drawTexture == 0) {
        color = ambient + diffuse + specular;
    } else if (drawTexture == 1) {

        vec4 texColor = texture2D(tex, textureVec.st);
        color = texColor  * (ambient + diffuse) + specular;

    } else if (drawTexture == 2) {
        color = texture2D(tex, textureVec.st);
    }  else if (drawTexture == -1){

        vec3 view_vec = normalize(vec3(vec4(0, 0, 0, 1) - eye_pos));
        vec3 direc = normalize(reflect(-view_vec, neg_norm));
        direc = vec3(unMatrix * vec4(direc, 0));
        color = textureCube(cube_texture, direc);
    } else if (drawTexture == -2) {
        // Normal mapping code is from https://apoorvaj.io/exploring-bump-mapping-with-webgl/
        vec3 light_dir = normalize(tang_light_pos - tang_frag_pos);
        vec3 view_dir = normalize(tang_eye_pos - tang_frag_pos);

        vec2 uv = textureVec.st;

        vec3 norm = normalize(texture2D(texnormal, uv).rgb * 2.0 - 1.0);
        float diffuse2 = max(dot(light_dir, norm), 0.0);
        vec3 tex_col = texture2D(tex, uv).rgb;


        color =  1.0 * vec4(diffuse2 * tex_col + 0.2 * tex_col, 1.0);

    }

    gl_FragColor = color;


}
`;

var vertexshadersource = `
// Based on color-triangles.html
precision highp float;
attribute vec3 aVertexPosition;
attribute vec3 aVertexColor;
attribute vec3 aVertexNorm;
attribute vec2 aTextureLoc;

attribute vec3 aVertexTang;
attribute vec3 aVertexBiTang;

varying vec4 vColor;

varying vec4 light_pos;
varying vec4 view_pos;
uniform vec4 ambient_mat;
uniform vec4 ambient_light;


uniform mat4 pMatrix;
uniform mat4 mvMatrix;

uniform mat4 mMatrix;
uniform mat4 vMatrix;

uniform int drawEdge;
uniform vec3 edgeColor;

varying vec3 norm_vec;
varying vec3 neg_norm;

varying vec3 w_norm;
varying vec4 eye_pos;
varying vec4 textureVec;

uniform vec3 light_loc;
uniform vec3 camera_pos;
uniform float shine_val;

uniform mat4 unMatrix;
uniform mat4 umMatrix;
uniform mat4 uv2wMatrix;

uniform int reverseNorm;
uniform int use2DTex;

varying vec3 tang_light_pos;
varying vec3 tang_eye_pos;
varying vec3 tang_frag_pos;

varying vec2 pass_bump;


uniform int useBumpTex;

void main(void) {
    // based on uniform-color-xform.html in the https://github.com/hguo/WebGL-tutorial.git to set the point size
    // and also 12-shading.html and 12-shading.js in the given tutorial
    gl_PointSize = 5.0;
    // mvMatrix = vMatrix * mMatrix and is the modelview Matrix
    // based on mario.js
    if (reverseNorm == 1) {
        norm_vec = normalize(vec3(unMatrix * vec4(-aVertexNorm, 0.0)));
    } else {
        norm_vec = normalize(vec3(unMatrix * vec4(aVertexNorm, 0.0)));
    }

    neg_norm = normalize(vec3(unMatrix * vec4(-aVertexNorm, 0.0)));

    // view_pos is based on https://webglfundamentals.org/webgl/lessons/webgl-environment-maps.html
    w_norm = normalize(vec3(unMatrix * vec4(aVertexNorm, 0.0)));
    view_pos = mMatrix * vec4(aVertexPosition, 1.0);
    light_pos = vMatrix * mMatrix * vec4(light_loc, 1.0);
    eye_pos = vMatrix * mMatrix * vec4(aVertexPosition, 1.0);
    gl_Position = pMatrix * vMatrix * mMatrix * vec4(aVertexPosition, 1.0);

    // adapted from the mario-uv-per-frag.html
    textureVec = vec4(0, 0, 0, 1);

    if (use2DTex == 1) {
        textureVec = vec4(aTextureLoc.st, 0.0, 1.0);
    }

    pass_bump = vec2(10.0, 10.0);
    tang_light_pos = vec3(0, 0, 0);
    tang_eye_pos = vec3(0, 0, 0);
    tang_frag_pos = vec3(0, 0, 0);

    if (useBumpTex == 0) {
        return;
    }

    if (useBumpTex == 1) {
        // The shader code adaped a lot from the website https://apoorvaj.io/exploring-bump-mapping-with-webgl/
        textureVec = vec4(aTextureLoc.st, 0.0, 1.0);
        vec3 vertexNorms = cross(aVertexTang, aVertexBiTang);
        //vertexNorms = neg_norm;

        vec3 t = normalize(vec3(unMatrix * vec4(aVertexTang, 0.0)));
        vec3 b = normalize(vec3(unMatrix * vec4(aVertexBiTang, 0.0)));
        vec3 n = normalize(vec3(unMatrix * vec4(vertexNorms, 0.0)));
        mat3 tbn = mat3(t, b, n);

        mat3 transTBN = mat3(tbn[0].x, tbn[1].x, tbn[2].x,
                             tbn[0].y, tbn[1].y, tbn[2].y,
                             tbn[0].z, tbn[1].z, tbn[2].z);

        tang_light_pos = transTBN * vec3(light_pos);
        tang_eye_pos = transTBN * vec3(-eye_pos);
        tang_frag_pos = transTBN * vec3(mMatrix * vec4(aVertexPosition, 1.0));
        pass_bump = vec2(0.0, 0.0);
   }



}
`;

function mycreateshader(gl, prog, source, type){
    var shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    // Based on shader_setups.js
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(shader));
    }
    gl.attachShader(prog, shader);
    return shader;
}

function initShaders() {
    shaderProgram = gl.createProgram();

    var fragmentShader = mycreateshader(gl, shaderProgram, fragshadersource, gl.FRAGMENT_SHADER);
    var vertexShader = mycreateshader(gl, shaderProgram, vertexshadersource, gl.VERTEX_SHADER);

    gl.linkProgram(shaderProgram);
    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        alert("Cannot load shader");
    }
    gl.useProgram(shaderProgram);
    shaderProgram.vertexPositionAttribute = gl.getAttribLocation(shaderProgram, "aVertexPosition");
    gl.enableVertexAttribArray(shaderProgram.vertexPositionAttribute);

    shaderProgram.vertexTangAttribute = gl.getAttribLocation(shaderProgram, "aVertexTang");
    gl.enableVertexAttribArray(shaderProgram.vertexTangAttribute);

    shaderProgram.vertexBiTangAttribute = gl.getAttribLocation(shaderProgram, "aVertexBiTang");
    gl.enableVertexAttribArray(shaderProgram.vertexBiTangAttribute);

    // Based on color-triangles.js
    //shaderProgram.vertexColorAttribute = gl.getAttribLocation(shaderProgram, "aVertexColor");
    //gl.enableVertexAttribArray(shaderProgram.vertexColorAttribute);

    shaderProgram.vertexNormalAttribute = gl.getAttribLocation(shaderProgram, "aVertexNorm");
    gl.enableVertexAttribArray(shaderProgram.vertexNormalAttribute);

    shaderProgram.mvMatrixLoc = gl.getUniformLocation(shaderProgram, "mvMatrix");
    shaderProgram.pMatrixLoc = gl.getUniformLocation(shaderProgram, "pMatrix");
    shaderProgram.mMatLoc = gl.getUniformLocation(shaderProgram, "mMatrix");
    shaderProgram.vMatLoc = gl.getUniformLocation(shaderProgram, "vMatrix");

    shaderProgram.textureLocAttr = gl.getAttribLocation(shaderProgram, "aTextureLoc");
    gl.enableVertexAttribArray(shaderProgram.textureLocAttr);
}

function normalize_uv(uv_vertices) {
    // Make the value between 0 and 1
    var min_val = uv_vertices[0];
    var max_val = uv_vertices[0];

    for (var i = 2; i < uv_vertices.length; i += 2) {
        if (uv_vertices[i] > max_val) {
            max_val = uv_vertices[i];
        }
        if (uv_vertices[i] < min_val) {
            min_val = uv_vertices[i];
        }
    }

    for (var i = 2; i < uv_vertices.length; i += 2) {
        uv_vertices[i] = (uv_vertices[i] - min_val)/(max_val - min_val);
    }


    var min_val = uv_vertices[1];
    var max_val = uv_vertices[1];
    for (var i = 1; i < uv_vertices.length; i += 2) {
        if (uv_vertices[i] > max_val) {
            max_val = uv_vertices[i];
        }
        if (uv_vertices[i] < min_val) {
            min_val = uv_vertices[i];
        }
    }

    for (var i = 1; i < uv_vertices.length; i += 2) {
        uv_vertices[i] = (uv_vertices[i] - min_val)/(max_val - min_val);
    }
}

var surfaceVertexBuffer;
var surfaceVertices;

function initPoly(vertices2d, depth, color, rotDeg, scale, center, name) {
    // color is 3 elements array
    // rotDeg is 3 elements array
    var vertices = [];
    var vertices_tang = [];
    var vertices_bitang = [];
    var vertices_uv = [];

    var indices = [];
    console.log(vertices2d);

    var rotMat = mat4.identity(mat4.create());
    rotMat = mat4.rotateX(rotMat, rotDeg[0]);
    rotMat = mat4.rotateY(rotMat, rotDeg[1]);
    rotMat = mat4.rotateZ(rotMat, rotDeg[2]);
    var scaleMat = mat4.scale(mat4.identity(mat4.create()), scale);
    var trMat = mat4.translate(mat4.identity(mat4.create()), center);

    var pos_norm = [0, 0, 1];
    var neg_norm = [0, 0, -1];

    pos_norm = mat4.multiplyVec3(rotMat, pos_norm);
    neg_norm = mat4.multiplyVec3(rotMat, neg_norm);

    for (var i = 0; i < vertices2d.length - 1; i += 2) {
        vertices.push(vertices2d[i]);
        vertices.push(vertices2d[i + 1]);
        vertices.push(0);
        Array.prototype.push.apply(vertices, neg_norm);

        Array.prototype.push.apply(vertices_tang, [-vertices2d[i], -vertices2d[i + 1], 0]);
        Array.prototype.push.apply(vertices_bitang, [-vertices2d[i], 1 - vertices2d[i + 1], 0]);
        Array.prototype.push.apply(vertices_uv, [vertices2d[i], vertices2d[i + 1]]);

        vertices.push(vertices2d[i]);
        vertices.push(vertices2d[i + 1]);
        vertices.push(depth);
        Array.prototype.push.apply(vertices, pos_norm);

        Array.prototype.push.apply(vertices_tang, [-vertices2d[i], -vertices2d[i + 1], -depth]);
        Array.prototype.push.apply(vertices_bitang, [-vertices2d[i], 1 - vertices2d[i + 1], -depth]);
        Array.prototype.push.apply(vertices_uv, [vertices2d[i], vertices2d[i + 1]]);
    }


    for (var i = 0; i < vertices2d.length - 4; i += 2) {
        //if (i < vertices2d.length - 4) {
            indices.push(0);
            indices.push(i + 2);
            indices.push(i + 4);

            indices.push(1);
            indices.push(i + 3);
            indices.push(i + 5);
        //}

    }

    normalize_uv(vertices_uv);


    for (var i = 0; i < vertices2d.length - 2; i += 2) {
        vertices.push(vertices2d[i]);
        vertices.push(vertices2d[i + 1]);
        vertices.push(0);

        // Check https://www.storyofmathematics.com/orthogona-vector/
        // to get the orthogonal vector
        var surface_vec = [vertices2d[i] - vertices2d[i + 2], vertices2d[i + 1] - vertices2d[i + 3]];

        var norm_vec =  [surface_vec[1], -surface_vec[0], 0];

        var tang_vec = [surface_vec[0], surface_vec[1], 0];
        var bitang_vec = [0, 0, 1]; // An example
        norm_vec = mat4.multiplyVec3(rotMat, norm_vec);

        tang_vec = mat4.multiplyVec3(rotMat, tang_vec);
        bitang_vec = mat4.multiplyVec3(rotMat, bitang_vec);

        Array.prototype.push.apply(vertices, norm_vec);
        Array.prototype.push.apply(vertices_tang, tang_vec);
        Array.prototype.push.apply(vertices_bitang, bitang_vec);
        Array.prototype.push.apply(vertices_uv, [0, 0]);


        vertices.push(vertices2d[i]);
        vertices.push(vertices2d[i + 1]);
        vertices.push(depth);
        Array.prototype.push.apply(vertices, norm_vec);
        Array.prototype.push.apply(vertices_tang, tang_vec);
        Array.prototype.push.apply(vertices_bitang, bitang_vec);
        Array.prototype.push.apply(vertices_uv, [0, 1]);

        vertices.push(vertices2d[i + 2]);
        vertices.push(vertices2d[i + 3]);
        vertices.push(0);
        Array.prototype.push.apply(vertices, norm_vec);
        Array.prototype.push.apply(vertices_tang, tang_vec);
        Array.prototype.push.apply(vertices_bitang, bitang_vec);
        Array.prototype.push.apply(vertices_uv, [1, 1]);

        vertices.push(vertices2d[i + 2]);
        vertices.push(vertices2d[i + 3]);
        vertices.push(depth);
        Array.prototype.push.apply(vertices, norm_vec);
        Array.prototype.push.apply(vertices_tang, tang_vec);
        Array.prototype.push.apply(vertices_bitang, bitang_vec);
        Array.prototype.push.apply(vertices_uv, [1, 0]);

        indices.push(vertices2d.length + 2 * i);
        indices.push(vertices2d.length + 2 * i + 1);
        indices.push(vertices2d.length + 2 * i + 2);

        indices.push(vertices2d.length + 2 * i + 1);
        indices.push(vertices2d.length + 2 * i + 2);
        indices.push(vertices2d.length + 2 * i + 3);

    }

    vertices.push(vertices2d[vertices2d.length - 2]);
    vertices.push(vertices2d[vertices2d.length - 1]);
    vertices.push(0);

    // Check https://www.storyofmathematics.com/orthogona-vector/
    // to get the orthogonal vector
    var surface_vec = [vertices2d[vertices2d.length - 2] - vertices2d[0], vertices2d[vertices2d.length - 1] - vertices2d[1]];

    var tang_vec = [surface_vec[0], surface_vec[1], 0];
    var bitang_vec = [0, 0, 1]; // An example
    var norm_vec =  [surface_vec[1], -surface_vec[0], 0];
    norm_vec = mat4.multiplyVec3(rotMat, norm_vec);
    tang_vec = mat4.multiplyVec3(rotMat, tang_vec);
    bitang_vec = mat4.multiplyVec3(rotMat, bitang_vec);

    Array.prototype.push.apply(vertices, norm_vec);
    Array.prototype.push.apply(vertices_tang, tang_vec);
    Array.prototype.push.apply(vertices_bitang, bitang_vec);
    Array.prototype.push.apply(vertices_uv, [0, 0]);

    vertices.push(vertices2d[vertices2d.length - 2]);
    vertices.push(vertices2d[vertices2d.length - 1]);
    vertices.push(depth);
    Array.prototype.push.apply(vertices, norm_vec);
    Array.prototype.push.apply(vertices_tang, tang_vec);
    Array.prototype.push.apply(vertices_bitang, bitang_vec);
    Array.prototype.push.apply(vertices_uv, [1, 1]);

    vertices.push(vertices2d[0]);
    vertices.push(vertices2d[1]);
    vertices.push(0);
    Array.prototype.push.apply(vertices, norm_vec);
    Array.prototype.push.apply(vertices_tang, tang_vec);
    Array.prototype.push.apply(vertices_bitang, bitang_vec);
    Array.prototype.push.apply(vertices_uv, [0, 1]);

    vertices.push(vertices2d[0]);
    vertices.push(vertices2d[1]);
    vertices.push(depth);
    Array.prototype.push.apply(vertices, norm_vec);
    Array.prototype.push.apply(vertices_tang, tang_vec);
    Array.prototype.push.apply(vertices_bitang, bitang_vec);
    Array.prototype.push.apply(vertices_uv, [1, 0]);

    //console.log("now vertices "  + name + "  "  + vertices.length + " " + vertices2d.length);

    indices.push(3 * vertices2d.length - 4);
    indices.push(3 * vertices2d.length - 3);
    indices.push(3 * vertices2d.length - 2);

    indices.push(3 * vertices2d.length - 3);
    indices.push(3 * vertices2d.length - 2);
    indices.push(3 * vertices2d.length - 1);


    applyVertices(rotMat, vertices, 6);
    applyVertices(scaleMat, vertices, 6);
    applyVertices(trMat, vertices, 6);

    //console.log("check length");
    //console.log(vertices.length);
    //console.log(vertices_tang.length);
    //console.log(vertices_bitang.length);
    //console.log(vertices_uv.length);
    initVBO(name, vertices, indices, color, vertices_tang, vertices_bitang, vertices_uv);

}

function initCube(size, color, center, name) {
    var [vertices, indices, edges, norms, tangs, bitangs, uvs] = getCubeVertices(size, color, center);
    //var norms = getTriangleNorms(vertices, indices, 3);



    initVBOwNorm(name, vertices, vertices.length, indices, indices.length, norms, color, edges, tangs, bitangs, uvs);

    /*
     *
    //       7       4
    //   3      0
    //      6       5
    //   2      1
     *     var indices = [0, 2, 1, 0, 3, 2,
                   0, 1, 5, 0, 5, 4,
                   1, 2, 6, 1, 6, 5,
                   4, 5, 6, 4, 6, 7,
                   3, 0, 4, 3, 7, 4,
                   2, 7, 3, 2, 6, 7
                  ];
     */
}


function getTriangleNorms(vertices, indices, stride) {
// Based on computeVertexNormals in mario.js
    var norms = [];
    //console.log(vertices);
    for (var i = 0; i < indices.length; i += 3) {
        var p0 = [vertices[stride * indices[i]], vertices[stride * indices[i] + 1], vertices[stride * indices[i] + 2]];
        var p1 = [vertices[stride * indices[i + 1]], vertices[stride * indices[i + 1] + 1], vertices[stride * indices[i + 1] + 2]];
        var p2 = [vertices[stride * indices[i + 2]], vertices[stride * indices[i + 2] + 1], vertices[stride * indices[i + 2] + 2]];

        var x = [p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2]];
        var y = [p0[0] - p2[0], p0[1] - p2[1], p0[2] - p2[2]];

        norms.push(x[1] * y[2] - x[2] * y[1]);
        norms.push(x[0] * y[2] - x[2] * y[0]);
        norms.push(x[0] * y[1] - x[1] * y[0]);


        /*if (i == 0) {
            console.log(p0);
            console.log(p1);
            console.log(p2);
            console.log("check p0");
            console.log(p0[0] + " " + p0[1] + " " + p0[2]);
            console.log("check x");
            console.log(x[0] + " " + x[1] + " " + x[2]);
            console.log("check norm");
            console.log(norms[norms.length - 3] + " " + norms[norms.length - 2] + " " + norms[norms.length - 1]);
        }*/

    }

    return norms;
}

function computeNorm(p0, p1, p2) {
    // Adapted based on computeSurfaceNormals  and   computeVertexNormals
    // in mario.js in the given tutorial
    var v1 = [p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2]];
    var v2 = [p0[0] - p2[0], p0[1] - p2[1], p0[2] - p2[2]];

    var norm = [
        v1[1] * v2[2] - v2[1] * v1[2],
        v1[2] * v2[0] - v2[2] * v1[0],
        v1[0] * v2[1] - v2[0] * v1[1]
    ];


    return norm;
}

var jsondata;
var jsonready = 0;

function initFromJson(file_name) {
    // Code adapted from the mario.js in the given tutorial
    let request = new XMLHttpRequest();
    request.open("GET", file_name);
    // Based on MDN https://developer.mozilla.org/en-US/docs/Web/API/XMLHttpRequest/readystatechange_event

    request.onreadystatechange = () => {
        if (request.readyState == 4) {
            jsondata = JSON.parse(request.responseText);
            //var verticesToFace = new Map();
            // The code is based on  computeSurfaceNormals and
            // computeVertexNormals in mario.js

            var indices = [];
            var facesNorms = [];
            for (var i = 0; i < jsondata.faces.length; i += 11) {
                indices.push(jsondata.faces[i + 1]);
                indices.push(jsondata.faces[i + 2]);
                indices.push(jsondata.faces[i + 3]);

                var facenorm = computeNorm([jsondata.vertices[3 * jsondata.faces[i + 1]],  jsondata.vertices[3 * jsondata.faces[i + 1] + 1], jsondata.vertices[3 * jsondata.faces[i + 1] + 2]],
                                           [jsondata.vertices[3 * jsondata.faces[i + 2]],  jsondata.vertices[3 * jsondata.faces[i + 2] + 1], jsondata.vertices[3 * jsondata.faces[i + 2] + 2]],
                                           [jsondata.vertices[3 * jsondata.faces[i + 3]],  jsondata.vertices[3 * jsondata.faces[i + 3] + 1], jsondata.vertices[3 * jsondata.faces[i + 3] + 2]]);

                facesNorms.push(facenorm);
            }

            //console.log(facesNorms);

            var norms = new Float32Array(jsondata.vertices.length);

            for (var i = 0; i < facesNorms.length; ++i) {
                var tri0 = indices[3 * i];
                var tri1 = indices[3 * i + 1];
                var tri2 = indices[3 * i + 2];


                norms[3 * indices[3 * i]] += facesNorms[i][0];
                norms[3 * indices[3 * i] + 1] += facesNorms[i][1];
                norms[3 * indices[3 * i] + 2] += facesNorms[i][2];

                norms[3 * indices[3 * i + 1]] += facesNorms[i][0];
                norms[3 * indices[3 * i + 1] + 1] += facesNorms[i][1];
                norms[3 * indices[3 * i + 1] + 2] += facesNorms[i][2];

                norms[3 * indices[3 * i + 2]] += facesNorms[i][0];
                norms[3 * indices[3 * i + 2] + 1] += facesNorms[i][1];
                norms[3 * indices[3 * i + 2] + 2] += facesNorms[i][2];

            }

            for (var i = 0; i < norms.length; i+= 3) {
                var normsum = Math.sqrt(norms[i] ** 2 + norms[i + 1] ** 2 + norms[i + 2] ** 2);
                norms[i] = norms[i] / normsum;
                norms[i + 1] = norms[i + 1] / normsum;
                norms[i + 2] = norms[i + 2] / normsum;
            }

            var scale = mat4.scale(mat4.identity(mat4.create()), [1, 1, 1]);

            var trans = mat4.translate(mat4.identity(mat4.create()), [-2, -0.1, 2.0]);


            applyVertices(scale, jsondata.vertices, 3);
            applyVertices(trans, jsondata.vertices, 3);

            //initVBOwNorm("car", jsondata.vertices, jsondata.vertices.length / 3, indices, indices.length, norms, [0, 0.9, 0]);
            initVBOwNormATex("car", jsondata.vertices, jsondata.vertices.length / 3, indices, indices.length, norms,
                             jsondata.materials[0].diffuse,
                             jsondata.materials[0].specular,
                             "static/images/car.jpg", jsondata.uvs[0]);

            jsonready = 1;
            drawScene();
        }
    };

    request.send();

}

function initVBOwNormATex(name, vertices, verticeNum, indices, indiceNum, norms,
                          diffuse_mat, specular_mat, texPic, texLoc) {
    var vbo = gl.createBuffer();
    vboMap.set(name, vbo);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    vbo.numItems = verticeNum;
    vbo.vertices = vertices;
    vbo.diffuse_mat = diffuse_mat;
    vbo.specular_mat = specular_mat;

    var normVBO = gl.createBuffer();
    vboMap.set(name + "_norm", normVBO);
    gl.bindBuffer(gl.ARRAY_BUFFER, normVBO);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(norms), gl.STATIC_DRAW);
    normVBO.numItems = norms.length / 3;
    normVBO.norms = norms;

    var indVBO = gl.createBuffer();
    vboMap.set(name + "_ind", indVBO);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indVBO);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    indVBO.numItems = indiceNum;
    indVBO.indices = indices;

    var texVBO = gl.createBuffer();
    vboMap.set(name + "_texloc", texVBO);
    // Adapted code based on mario-uv.js
    gl.bindBuffer(gl.ARRAY_BUFFER, texVBO);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(texLoc), gl.STATIC_DRAW);
    texVBO.numItems = texLoc.length / 2;
    texVBO.texLoc = texLoc;

    var textureobj = load2dtexturefromfile(texPic);
    vboMap.set(name + "_texobj", textureobj);
}

function load2dtexturefromfile(texPic) {
    // Following code adapted based on mario-uv.js in the Given tutorial
    // which also cites  https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Using_textures_in_WebGL
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);

    gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        1,  // width
        1,  // height
        0,  // border
        gl.RGBA,   // source Format
        gl.UNSIGNED_BYTE,    // source Type
        new Uint8Array([0, 255, 255, 255])
    );


    const image = new Image();
    image.src = texPic;

    image.onload = () => {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);


        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA, // internalFormat,
            gl.RGBA, // srcType,
            gl.UNSIGNED_BYTE,
            image
        );

        //gl.generateMipmap(gl.TEXTURE_2D);
        // Based on textureTeapot.js
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.bindTexture(gl.TEXTURE_2D, null);

        drawScene();
    };
    return texture;

}

// cubemap code is based on Carmen
// and also https://webglfundamentals.org/webgl/lessons/webgl-cube-maps.html
function initCubeMapTexture() {
    var cubemapTexture = gl.createTexture();

    var images = [];
    const image_names = ["static/images/negx.jpg", "static/images/negy.jpg", "static/images/negz.jpg", "static/images/posx.jpg", "static/images/posy.jpg", "static/images/posz.jpb"];
    const gl_cube_names = [gl.TEXTURE_CUBE_MAP_NEGATIVE_X,
                           gl.TEXTURE_CUBE_MAP_NEGATIVE_Y,
                           gl.TEXTURE_CUBE_MAP_NEGATIVE_Z,
                           gl.TEXTURE_CUBE_MAP_POSITIVE_X,
                           gl.TEXTURE_CUBE_MAP_POSITIVE_Y,
                           gl.TEXTURE_CUBE_MAP_POSITIVE_Z
                          ];
    var colors = [
        new Uint8Array([255, 255, 0, 255]),
        new Uint8Array([0, 255, 255, 255]),
        new Uint8Array([255, 0, 255, 255]),
        new Uint8Array([0, 255, 0, 255]),
        new Uint8Array([0, 0, 255, 255]),
        new Uint8Array([255, 0, 0, 255])
    ];
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, cubemapTexture);
    for (var i = 0; i < 6; ++i) {
        // Some of environment map code are based on https://webglfundamentals.org/webgl/lessons/webgl-environment-maps.html
        // Also see https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Using_textures_in_WebGL
        // and mario-uv.js
        gl.bindTexture(gl.TEXTURE_CUBE_MAP, cubemapTexture);
        console.log(" i of 6 " + colors[i]);
        gl.texImage2D(gl_cube_names[i],
                      0,
                      gl.RGBA,
                      1,
                      1,
                      0,
                      gl.RGBA,
                      gl.UNSIGNED_BYTE,
                      colors[i]);

    }

    // Based on cubeMappedTeapot.js in the given tutorial
    cubemapTexture.imageposz = new Image();
    cubemapTexture.imageposz.src = "static/images/poszenv.jpg";

    cubemapTexture.imagenegz = new Image();
    cubemapTexture.imagenegz.src = "static/images/negzenv.jpg";

    cubemapTexture.imageposy = new Image();
    cubemapTexture.imageposy.src = "static/images/posyenv.jpg";

    cubemapTexture.imagenegy = new Image();
    cubemapTexture.imagenegy.src = "static/images/negyenv.jpg";

    cubemapTexture.imageposx = new Image();
    cubemapTexture.imageposx.src = "static/images/posxenv.jpg";

    cubemapTexture.imagenegx = new Image();
    cubemapTexture.imagenegx.src = "static/images/negxenv.jpg";

    cubemapTexture.imageposz.onload = () => {
        gl.bindTexture(gl.TEXTURE_CUBE_MAP, cubemapTexture);

        // Set parameter based on cubeMappedTeapot.js
        // Flip based on  https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext/pixelStorei

        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);


        // Switch x direction
        gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cubemapTexture.imageposx);
        gl.texImage2D(gl.TEXTURE_CUBE_MAP_NEGATIVE_X, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cubemapTexture.imagenegx);

        gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_Y, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cubemapTexture.imageposy);
        gl.texImage2D(gl.TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cubemapTexture.imagenegy);


        gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_Z, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cubemapTexture.imageposz);
        gl.texImage2D(gl.TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cubemapTexture.imagenegz);


    };


    vboMap.set("cubeTexobj", cubemapTexture);

}

function initVBO(name, vertices, indices, color, tang_vecs, bitang_vecs, uv_vecs) {
    var vbo = gl.createBuffer();
    vboMap.set(name, vbo);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    vbo.numItems = vertices.length / 6;
    vbo.vertices = vertices;
    vbo.color = color;
    vbo.diffuse_mat = color;
    vbo.specular_mat = specular_mat;

    var indVBO = gl.createBuffer();
    vboMap.set(name + "_ind", indVBO);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indVBO);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    indVBO.numItems = indices.length;
    indVBO.indices = indices;

    if (tang_vecs && bitang_vecs && uv_vecs) {
        //console.log("check exits!")
        var tangVBO = gl.createBuffer();
        vboMap.set(name + "_tang", tangVBO);
        gl.bindBuffer(gl.ARRAY_BUFFER, tangVBO);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(tang_vecs), gl.STATIC_DRAW);

        var bitangVBO = gl.createBuffer();
        vboMap.set(name + "_bitang", bitangVBO);
        gl.bindBuffer(gl.ARRAY_BUFFER, bitangVBO);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(bitang_vecs), gl.STATIC_DRAW);

        var texlocVBO = gl.createBuffer();
        vboMap.set(name + "_texloc", texlocVBO);
        gl.bindBuffer(gl.ARRAY_BUFFER, texlocVBO);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(uv_vecs), gl.STATIC_DRAW);

        var tex_diffuse = load2dtexturefromfile("static/images/bump_diffuse.png");
        //var tex_depth = load2dtexturefromfile("bump_depth.png");
        var tex_normal = load2dtexturefromfile("static/images/bump_normal.png");

        vboMap.set(name + "_texdiffuse", tex_diffuse);
        //vboMap.set(name + "_texdepth", tex_depth);
        vboMap.set(name + "_texnormal", tex_normal);

    }

}

function initVBOwNorm(name, vertices, verNums, indices, indNums, norms, color, edges, tang_vecs, bitang_vecs, uv_vecs) {
    var vbo = gl.createBuffer();
    vboMap.set(name, vbo);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    vbo.numItems = verNums;
    vbo.vertices = vertices;
    vbo.color = color;
    vbo.diffuse_mat = color;
    vbo.specular_mat = specular_mat;

    var normVBO = gl.createBuffer();
    vboMap.set(name + "_norm", normVBO);
    gl.bindBuffer(gl.ARRAY_BUFFER, normVBO);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(norms), gl.STATIC_DRAW);
    normVBO.numItems = norms.length / 3;
    normVBO.norms = norms;

    var indVBO = gl.createBuffer();
    vboMap.set(name + "_ind", indVBO);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indVBO);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    indVBO.numItems = indNums;
    indVBO.indices = indices;

    //var texLocVBO = gl.createBuffer();
    //vboMap.set(name + "_texloc", texLocVBO);
    //gl.bindBuffer(gl.ARRAY_BUFFER, texLocVBO);
    //gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(2 * verNums), gl.STATIC_DRAW);

    if (edges) {
        var edgeVBO = gl.createBuffer();
        vboMap.set(name + "_edge", edgeVBO);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, edgeVBO);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(edges), gl.STATIC_DRAW);
        edgeVBO.numItems = edges.length;
        edgeVBO.edges = edges;
    }

    if (tang_vecs && bitang_vecs && uv_vecs) {
        //console.log("check exits!")
        var tangVBO = gl.createBuffer();
        vboMap.set(name + "_tang", tangVBO);
        gl.bindBuffer(gl.ARRAY_BUFFER, tangVBO);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(tang_vecs), gl.STATIC_DRAW);

        var bitangVBO = gl.createBuffer();
        vboMap.set(name + "_bitang", bitangVBO);
        gl.bindBuffer(gl.ARRAY_BUFFER, bitangVBO);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(bitang_vecs), gl.STATIC_DRAW);

        var texlocVBO = gl.createBuffer();
        vboMap.set(name + "_texloc", texlocVBO);
        gl.bindBuffer(gl.ARRAY_BUFFER, texlocVBO);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(uv_vecs), gl.STATIC_DRAW);

        var tex_diffuse = load2dtexturefromfile("static/images/bump_diffuse.png");
        //var tex_depth = load2dtexturefromfile("bump_depth.png");
        var tex_normal = load2dtexturefromfile("static/images/bump_normal.png");

        vboMap.set(name + "_texdiffuse", tex_diffuse);
        //vboMap.set(name + "_texdepth", tex_depth);
        vboMap.set(name + "_texnormal", tex_normal);

    }
}

function getCubeVertices(size, color, center){
    var vertices = [
        center[0] + size/2, center[1] + size/2, center[2] + size/2,
        center[0] + size/2, center[1] - size/2, center[2] + size/2,
        center[0] - size/2, center[1] - size/2, center[2] + size/2,
        center[0] - size/2, center[1] + size/2, center[2] + size/2,
        center[0] + size/2, center[1] + size/2, center[2] - size/2,
        center[0] + size/2, center[1] - size/2, center[2] - size/2,
        center[0] - size/2, center[1] - size/2, center[2] - size/2,
        center[0] - size/2, center[1] + size/2, center[2] - size/2,

        center[0] + size/2, center[1] + size/2, center[2] + size/2,
        center[0] - size/2, center[1] + size/2, center[2] + size/2,
        center[0] - size/2, center[1] + size/2, center[2] - size/2,
        center[0] + size/2, center[1] + size/2, center[2] - size/2,

        center[0] + size/2, center[1] - size/2, center[2] + size/2,
        center[0] - size/2, center[1] - size/2, center[2] + size/2,
        center[0] - size/2, center[1] - size/2, center[2] - size/2,
        center[0] + size/2, center[1] - size/2, center[2] - size/2,

        center[0] + size/2, center[1] + size/2, center[2] + size/2,
        center[0] + size/2, center[1] - size/2, center[2] + size/2,
        center[0] + size/2, center[1] - size/2, center[2] - size/2,
        center[0] + size/2, center[1] + size/2, center[2] - size/2,

        center[0] - size/2, center[1] + size/2, center[2] + size/2,
        center[0] - size/2, center[1] - size/2, center[2] + size/2,
        center[0] - size/2, center[1] - size/2, center[2] - size/2,
        center[0] - size/2, center[1] + size/2, center[2] - size/2,
    ];

    //       10      11
    //   9        8

    //       14          15
    //   13         12

    //       7       4
    //   3      0
    //      6       5
    //   2      1
    // The index is based on adapted from the 3Dcube.js from the given tutorial
    //     6     7
    //   5    4
    //     2     3
    //   1    0
    var indices = [0, 3, 1, 1, 3, 2,
                   4, 7, 5, 7, 6, 5,
                   8, 9, 10, 8, 10, 11,
                   12, 13, 14, 12, 15, 14,
                   16, 17, 18, 16, 18, 19,
                   20, 21, 22, 20, 22, 23
                  ];

    var norms = [ 0, 0, 1, 0, 0, 1,
                  0, 0, 1, 0, 0, 1,
                   0, 0, -1, 0, 0, -1,
                    0, 0, -1, 0, 0, -1,
                 0, 1, 0, 0, 1, 0,
                 0, 1, 0, 0, 1, 0,
                 0, -1, 0, 0, -1, 0,
                 0, -1, 0, 0, -1, 0,
                 1, 0, 0, 1, 0, 0,
                 1, 0, 0, 1, 0, 0,
                 -1, 0, 0, -1, 0, 0,
                 -1, 0, 0, -1, 0, 0
                ];


    var edges = [0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 3, 7, 2, 6, 1, 5, 4, 5, 5, 6, 6, 7, 7, 4];

    var tangs = [0, -1, 0, 0, -1, 0,
                 0, -1, 0, 0, -1, 0,
                 0, -1, 0, 0, -1, 0,
                 0, -1, 0, 0, -1, 0,

                 0, 0, 1, 0, 0, 1,
                 0, 0, 1, 0, 0, 1,
                 0, 0, -1, 0, 0, -1,
                 0, 0, -1, 0, 0, -1,

                 0, -1, 0, 0, -1, 0,
                 0, -1, 0, 0, -1, 0,
                 0, -1, 0, 0, -1, 0,
                 0, -1, 0, 0, -1, 0


                ];

    var bitangs = [-1, 0, 0, -1, 0, 0,
                   -1, 0, 0, -1, 0, 0,
                   1, 0, 0, 1, 0, 0,
                   1, 0, 0, 1, 0, 0,

                   -1, 0, 0, -1, 0, 0,
                   -1, 0, 0, -1, 0, 0,
                   1, 0, 0, 1, 0, 0,
                   1,  0, 0, 1, 0, 0,

                   0, 0, 1, 0, 0, 1,
                   0, 0, 1, 0, 0, 1,
                   0, 0, -1, 0, 0, -1,
                   0, 0, -1, 0, 0, -1

                  ]

    var uvs = [0, 0, 2, 0, 2, 2, 0, 2,
               0, 0, 2, 0, 2, 2, 0, 2,
               0, 0, 2, 0, 2, 2, 0, 2,
               0, 0, 2, 0, 2, 2, 0, 2,
               0, 0, 2, 0, 2, 2, 0, 2,
               0, 0, 2, 0, 2, 2, 0, 2
               ];

    return [vertices, indices, edges, norms, tangs, bitangs, uvs];
}

function drawShapewBump(name) {
    var vbo = vboMap.get(name);

    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, 3, gl.FLOAT, false, 0, 0);

    var normvbo = vboMap.get(name + "_norm");
    gl.bindBuffer(gl.ARRAY_BUFFER, normvbo);
    gl.vertexAttribPointer(shaderProgram.vertexNormalAttribute, 3, gl.FLOAT, false, 0, 0);

    gl.uniform1i(gl.getUniformLocation(shaderProgram, "udrawTexture"), -2);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "useBumpTex"), 1);


    texvbo = vboMap.get(name + "_texloc");
    gl.bindBuffer(gl.ARRAY_BUFFER, texvbo);
    gl.vertexAttribPointer(shaderProgram.textureLocAttr, 2, gl.FLOAT, false, 0, 0);


    var tex_normal = vboMap.get(name + "_texnormal")
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex_normal);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "texnormal"), 0);


    var tex_diffuse = vboMap.get(name + "_texdiffuse")
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, tex_diffuse);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "tex"), 1);

    //var tex_depth = vboMap.get(name + "_texdepth")
    //gl.activeTexture(gl.TEXTURE7);
    //gl.bindTexture(gl.TEXTURE_2D, tex_depth);
    //gl.uniform1i(gl.getUniformLocation(shaderProgram, "texdepth"), 7);

    gl.bindBuffer(gl.ARRAY_BUFFER, vboMap.get(name + "_tang"))
    gl.vertexAttribPointer(shaderProgram.vertexTangAttribute, 3, gl.FLOAT, false, 0, 0);


    gl.bindBuffer(gl.ARRAY_BUFFER, vboMap.get(name + "_bitang"))
    gl.vertexAttribPointer(shaderProgram.vertexBiTangAttribute, 3, gl.FLOAT, false, 0, 0);


    indvbo = vboMap.get(name + "_ind");
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indvbo);
    gl.drawElements(gl.TRIANGLES, indvbo.numItems, gl.UNSIGNED_SHORT, 0);

    gl.uniform1i(gl.getUniformLocation(shaderProgram, "udrawTexture"), 0);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "useBumpTex"), 0);

}

function drawShape(name, drawType, reverseNorm) {
    vbo = vboMap.get(name);

    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, 3, gl.FLOAT, false, 24, 0);
    gl.vertexAttribPointer(shaderProgram.vertexNormalAttribute, 3, gl.FLOAT, false, 24, 12);
    gl.uniform4f(gl.getUniformLocation(shaderProgram, "diffuse_mat"), vbo.color[0], vbo.color[1], vbo.color[2], 1.0);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "udrawTexture"), 0);

    // This is fake as we actually do not use them
    gl.vertexAttribPointer(shaderProgram.vertexBiTangAttribute, 3, gl.FLOAT, false, 24, 12);
    gl.vertexAttribPointer(shaderProgram.vertexTangAttribute, 3, gl.FLOAT, false, 24, 12);
    gl.vertexAttribPointer(shaderProgram.textureLocAttr, 2, gl.FLOAT, false, 24, 0);

    if (reverseNorm == true) {
        gl.uniform1i(gl.getUniformLocation(shaderProgram, "reverseNorm"), 1);
    }

    indvbo = vboMap.get(name + "_ind");

    //console.log(name + "  rev norm " + reverseNorm);

    if (drawType == gl.POINTS) {
        gl.drawArrays(gl.POINTS, 0, vbo.numItems);
    } else if (drawType == gl.LINES) {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indvbo);
        gl.drawElements(gl.LINES, indvbo.numItems, gl.UNSIGNED_SHORT, 0);
    } else if (drawType == gl.TRIANGLES){
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indvbo);
        gl.drawElements(gl.TRIANGLES, indvbo.numItems, gl.UNSIGNED_SHORT, 0);
    }




    if (reverseNorm == true) {
        gl.uniform1i(gl.getUniformLocation(shaderProgram, "reverseNorm"), 0);
    }
}

function drawShapewTexture(name, flag) {
    vbo = vboMap.get(name);

    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, 3, gl.FLOAT, false, 0, 0);

    // This is fake as we actually do not use them
    gl.vertexAttribPointer(shaderProgram.vertexBiTangAttribute, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribPointer(shaderProgram.vertexTangAttribute, 3, gl.FLOAT, false, 0, 0);

    gl.uniform4f(gl.getUniformLocation(shaderProgram, "diffuse_mat"), vbo.diffuse_mat[0], vbo.diffuse_mat[1], vbo.diffuse_mat[2], 1.0);
    gl.uniform4f(gl.getUniformLocation(shaderProgram, "specular_mat"), vbo.specular_mat[0], vbo.specular_mat[1], vbo.specular_mat[2], 1.0);

    normvbo = vboMap.get(name + "_norm");
    gl.bindBuffer(gl.ARRAY_BUFFER, normvbo);
    gl.vertexAttribPointer(shaderProgram.vertexNormalAttribute, 3, gl.FLOAT, false, 0, 0);

    gl.uniform1i(gl.getUniformLocation(shaderProgram, "udrawTexture"), flag);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "use2DTex"), 1);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "useBumpTex"), 0);

    texvbo = vboMap.get(name + "_texloc");
    gl.bindBuffer(gl.ARRAY_BUFFER, texvbo);
    gl.vertexAttribPointer(shaderProgram.textureLocAttr, 2, gl.FLOAT, false, 0, 0);

    texObj = vboMap.get(name + "_texobj");
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "tex"), 0);
    gl.bindTexture(gl.TEXTURE_2D, texObj);

    indvbo = vboMap.get(name + "_ind");
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indvbo);
    //console.log("Check numitems  " + indvbo.numItems);
    gl.drawElements(gl.TRIANGLES, indvbo.numItems, gl.UNSIGNED_SHORT, 0);

    gl.uniform1i(gl.getUniformLocation(shaderProgram, "udrawTexture"), 0);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "use2DTex"), 0);
}

function drawShapeReflectTextureNoBump(name) {
    vbo = vboMap.get(name);

    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, 3, gl.FLOAT, false, 0, 0);

    // This is fake as we actually do not use them
    gl.vertexAttribPointer(shaderProgram.vertexBiTangAttribute, 3, gl.FLOAT, false, 24, 12);
    gl.vertexAttribPointer(shaderProgram.vertexTangAttribute, 3, gl.FLOAT, false, 24, 12);
    gl.vertexAttribPointer(shaderProgram.textureLocAttr, 2, gl.FLOAT, false, 0, 0);


    gl.uniform4f(gl.getUniformLocation(shaderProgram, "diffuse_mat"), vbo.diffuse_mat[0], vbo.diffuse_mat[1], vbo.diffuse_mat[2], 1.0);
    gl.uniform4f(gl.getUniformLocation(shaderProgram, "specular_mat"), vbo.specular_mat[0], vbo.specular_mat[1], vbo.specular_mat[2], 1.0);



    // Rotate based on the WebGL tutorial
    mat4.identity(mMatrix);
    //console.log('Z angle = '+ Z_angle);
    mMatrix = mat4.rotate(mMatrix, 0.0, [1, 1, 0]);
    gl.uniformMatrix4fv(shaderProgram.mMatLoc, false, mMatrix);

    normvbo = vboMap.get(name + "_norm");
    gl.bindBuffer(gl.ARRAY_BUFFER, normvbo);
    gl.vertexAttribPointer(shaderProgram.vertexNormalAttribute, 3, gl.FLOAT, false, 0, 0);

    texObj = vboMap.get("cubeTexobj");
    gl.activeTexture(gl.TEXTURE2);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "cube_texture"), 2);
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, texObj);

    gl.uniform1i(gl.getUniformLocation(shaderProgram, "udrawTexture"), -1);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "use2DTex"), 0);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "useBumpTex"), 0);


    indvbo = vboMap.get(name + "_ind");
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indvbo);
    //console.log("Check numitems  " + indvbo.numItems);
    gl.drawElements(gl.TRIANGLES, indvbo.numItems, gl.UNSIGNED_SHORT, 0);

    gl.uniform1i(gl.getUniformLocation(shaderProgram, "udrawTexture"), 0);

}

function drawShapeReflectTexture(name) {
    vbo = vboMap.get(name);

    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, 3, gl.FLOAT, false, 0, 0);
    gl.uniform4f(gl.getUniformLocation(shaderProgram, "diffuse_mat"), vbo.diffuse_mat[0], vbo.diffuse_mat[1], vbo.diffuse_mat[2], 1.0);
    gl.uniform4f(gl.getUniformLocation(shaderProgram, "specular_mat"), vbo.specular_mat[0], vbo.specular_mat[1], vbo.specular_mat[2], 1.0);


    // Rotate based on the WebGL tutorial
    mat4.identity(mMatrix);
    //console.log('Z angle = '+ Z_angle);
    mMatrix = mat4.rotate(mMatrix, 0.0, [1, 1, 0]);
    gl.uniformMatrix4fv(shaderProgram.mMatLoc, false, mMatrix);

    normvbo = vboMap.get(name + "_norm");
    gl.bindBuffer(gl.ARRAY_BUFFER, normvbo);
    gl.vertexAttribPointer(shaderProgram.vertexNormalAttribute, 3, gl.FLOAT, false, 0, 0);


    gl.uniform1i(gl.getUniformLocation(shaderProgram, "udrawTexture"), -2);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "use2DTex"), 1);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "useBumpTex"), 1);


    texvbo = vboMap.get(name + "_texloc");
    gl.bindBuffer(gl.ARRAY_BUFFER, texvbo);
    gl.vertexAttribPointer(shaderProgram.textureLocAttr, 2, gl.FLOAT, false, 0, 0);


    var tex_normal = vboMap.get(name + "_texnormal")
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, tex_normal);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "texnormal"), 1);


    var tex_diffuse = vboMap.get(name + "_texdiffuse")
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, tex_diffuse);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "tex"), 2);

    //var tex_depth = vboMap.get(name + "_texdepth")
    //gl.activeTexture(gl.TEXTURE7);
    //gl.bindTexture(gl.TEXTURE_2D, tex_depth);
    //gl.uniform1i(gl.getUniformLocation(shaderProgram, "texdepth"), 7);

    gl.bindBuffer(gl.ARRAY_BUFFER, vboMap.get(name + "_tang"))
    gl.vertexAttribPointer(shaderProgram.vertexTangAttribute, 3, gl.FLOAT, false, 0, 0);


    gl.bindBuffer(gl.ARRAY_BUFFER, vboMap.get(name + "_bitang"))
    gl.vertexAttribPointer(shaderProgram.vertexBiTangAttribute, 3, gl.FLOAT, false, 0, 0);


    indvbo = vboMap.get(name + "_ind");
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indvbo);
    gl.drawElements(gl.TRIANGLES, indvbo.numItems, gl.UNSIGNED_SHORT, 0);



    gl.uniform1i(gl.getUniformLocation(shaderProgram, "udrawTexture"), 0);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "use2DTex"), 0);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "useBumpTex"), 0);

}

function drawShapeReflectTextureCombNorm(name) {
    vbo = vboMap.get(name);

    texObj = vboMap.get("cubeTexobj");
    gl.activeTexture(gl.TEXTURE2);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "cube_texture"), 2);
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, texObj);

    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, 3, gl.FLOAT, false, 24, 0);
    gl.vertexAttribPointer(shaderProgram.vertexNormalAttribute, 3, gl.FLOAT, false, 24, 12);

    // This is fake as we actually do not use them
    gl.vertexAttribPointer(shaderProgram.vertexBiTangAttribute, 3, gl.FLOAT, false, 24, 12);
    gl.vertexAttribPointer(shaderProgram.vertexTangAttribute, 3, gl.FLOAT, false, 24, 12);
    gl.vertexAttribPointer(shaderProgram.textureLocAttr, 2, gl.FLOAT, false, 0, 0);

    gl.uniform4f(gl.getUniformLocation(shaderProgram, "diffuse_mat"), vbo.diffuse_mat[0], vbo.diffuse_mat[1], vbo.diffuse_mat[2], 1.0);
    gl.uniform4f(gl.getUniformLocation(shaderProgram, "specular_mat"), vbo.specular_mat[0], vbo.specular_mat[1], vbo.specular_mat[2], 1.0);


    // Rotate based on the WebGL tutorial
    mat4.identity(mMatrix);
    //console.log('Z angle = '+ Z_angle);
    mMatrix = mat4.rotate(mMatrix, 0.0, [1, 1, 0]);
    gl.uniformMatrix4fv(shaderProgram.mMatLoc, false, mMatrix);


    gl.uniform1i(gl.getUniformLocation(shaderProgram, "udrawTexture"), -1);


    indvbo = vboMap.get(name + "_ind");
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indvbo);
    //console.log("Check numitems  " + indvbo.numItems);
    gl.drawElements(gl.TRIANGLES, indvbo.numItems, gl.UNSIGNED_SHORT, 0);

    gl.uniform1i(gl.getUniformLocation(shaderProgram, "udrawTexture"), 0);
}

function drawShapewNorm(name, drawType, step) {
    vbo = vboMap.get(name);

    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, 3, gl.FLOAT, false, step, 0);

     // This is fake as we actually do not use them
    gl.vertexAttribPointer(shaderProgram.vertexBiTangAttribute, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribPointer(shaderProgram.vertexTangAttribute, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribPointer(shaderProgram.textureLocAttr, 2, gl.FLOAT, false, 0, 0);

    gl.uniform4f(gl.getUniformLocation(shaderProgram, "diffuse_mat"), vbo.color[0], vbo.color[1], vbo.color[2], 1.0);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "udrawTexture"), 0);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "use2DTex"), 0);

    normvbo = vboMap.get(name + "_norm");
    gl.bindBuffer(gl.ARRAY_BUFFER, normvbo);
    gl.vertexAttribPointer(shaderProgram.vertexNormalAttribute, 3, gl.FLOAT, false, 0, 0);

    indvbo = vboMap.get(name + "_ind");
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indvbo);
    //console.log("Check numitems  " + indvbo.numItems);
    gl.drawElements(drawType, indvbo.numItems, gl.UNSIGNED_SHORT, 0);

    //texvbo = vboMap.get(name + "_texloc");
    //gl.bindBuffer(gl.ARRAY_BUFFER, texvbo);
    //gl.vertexAttribPointer(shaderProgram.textureLocAttr, 2, gl.FLOAT, false, 0, 0);



}



function initSphere(radius, numSlices, numStacks, color, center, name) {
    var [sphereVertices, indices, edges] = getSphereVertices(radius, numSlices, numStacks, color, center);
    initVBO(name, sphereVertices, indices, color);

}


function getSphereVertices(radius, numSlices, numStacks, color, center) {
    var start = [center[0] + radius, center[1], center[2]];
    var vertices = [];

    var rotLatUp = mat4.translate(mat4.identity(mat4.create()), center);
    rotLatUp = mat4.rotateZ(rotLatUp, Math.PI/(2 * numStacks));
    rotLatUp = mat4.translate(rotLatUp, [-center[0], -center[1], -center[2]]);

    var rotLatDown = mat4.translate(mat4.identity(mat4.create()), center);
    rotLatDown = mat4.rotateZ(rotLatDown, -Math.PI/(2 * numStacks));
    rotLatDown = mat4.translate(rotLatDown, [-center[0], -center[1], -center[2]]);

    var point = [start[0], start[1], start[2]];

    Array.prototype.push.apply(vertices, point);
    Array.prototype.push.apply(vertices, [point[0] - center[0], point[1] - center[1], point[2] - center[2]]);

    for (var h = 1; h < numStacks; ++h) {
        point = mat4.multiplyVec3(rotLatUp, point);

        Array.prototype.push.apply(vertices, point);
        Array.prototype.push.apply(vertices, [point[0] - center[0], point[1] - center[1], point[2] - center[2]]);
    }

    point = [start[0], start[1], start[2]];
    for (var h = 1; h < numStacks; ++h) {
        point = mat4.multiplyVec3(rotLatDown, point);

        Array.prototype.push.apply(vertices, point);
        Array.prototype.push.apply(vertices, [point[0] - center[0], point[1] - center[1], point[2] - center[2]]);
    }

    // Now vertices has 2 * numStacks - 1 items

    var rotLong = mat4.translate(mat4.identity(mat4.create()), center);
    rotLong = mat4.rotateY(rotLong, 2 * Math.PI/numSlices);
    rotLong = mat4.translate(rotLong, [-center[0], -center[1], -center[2]]);

    for (var i = 0; i < numSlices - 1; ++i) {
        var startoff = i * (2 * numStacks - 1);
        for (var h = 0; h < 2 * numStacks - 1; ++h) {
            var point = [vertices[6 * (startoff + h)], vertices[6 * (startoff + h) + 1], vertices[6 * (startoff + h) + 2]];
            point = mat4.multiplyVec3(rotLong, point);

            Array.prototype.push.apply(vertices, point);
            Array.prototype.push.apply(vertices, [point[0] - center[0], point[1] - center[1], point[2] - center[2]]);
        }
    }

    Array.prototype.push.apply(vertices, [center[0], center[1] + radius, center[2]]);
    Array.prototype.push.apply(vertices, [0, 1, 0]);

    Array.prototype.push.apply(vertices, [center[0], center[1] - radius, center[2]]);
    Array.prototype.push.apply(vertices, [0, -1, 0]);

    var indices = [];
    var edges = [];

    // Examples of the indices when numSlice = 2 numStacks = 3
    //   10
    //  2  7    2
    //     1  6   1
    //       0   5  0
    //     3   8   3
    //   4  9    4
    //  11

    for (var i = 0; i < numSlices; ++i) {
        edges.push(i * (2 * numStacks - 1));
        if (i == numSlices - 1) {
            edges.push(0);
        } else {
            edges.push((i + 1) * (2 * numStacks - 1));
        }

        for (h = 0; h < numStacks - 1; ++h) {
            var startoff = i * (2 * numStacks - 1) + h;
            indices.push(startoff);
            indices.push(startoff + 1);
            if (i == numSlices - 1) {
                indices.push(h);
            } else {
                indices.push(startoff + (2 * numStacks - 1));
            }

            indices.push(startoff + 1);
            if (i == numSlices - 1) {
                indices.push(h);
                indices.push(h + 1);
            } else {
                indices.push(startoff + (2 * numStacks - 1));
                indices.push(startoff + (2 * numStacks - 1) + 1);
            }
        }

        var startoff = i * (2 * numStacks - 1);


        indices.push(startoff);
        indices.push(startoff + numStacks);
        if (i == numSlices - 1) {
            indices.push(0);
        } else {
            indices.push(startoff + (2 * numStacks - 1));
        }

        if (i == numSlices - 1) {
            indices.push(startoff + numStacks);
            indices.push(0);
            indices.push(numStacks);
        } else {
            indices.push(startoff + numStacks);
            indices.push(startoff + (2 * numStacks - 1));
            indices.push(startoff + (2 * numStacks - 1) + numStacks);
        }


        for (h = numStacks; h < 2 * numStacks - 2; ++h) {
            var startoff = i * (2 * numStacks - 1) + h;
            indices.push(startoff);
            indices.push(startoff + 1);
            if (i == numSlices - 1) {
                indices.push(h);
            } else {
                indices.push(startoff + (2 * numStacks - 1));
            }

            indices.push(startoff + 1);
            if (i == numSlices - 1) {
                indices.push(h);
                indices.push(h + 1);
            } else {
                indices.push(startoff + (2 * numStacks - 1));
                indices.push(startoff + (2 * numStacks - 1) + 1);
            }
        }
    }



    for (var i = 0; i < numSlices; ++i) {
        indices.push((2 * numStacks - 1) * numSlices);
        indices.push((2 * numStacks - 1) * i + numStacks - 1);

        if (i == numSlices - 1) {
            indices.push(numStacks - 1);
        } else {
            indices.push((2 * numStacks - 1) * (i + 1) + numStacks - 1);
        }

    }


    for (var i = 0; i < numSlices; ++i) {
        indices.push((2 * numStacks - 1) * numSlices + 1);
        indices.push((2 * numStacks - 1) * i + 2 * numStacks - 2);

        if (i == numSlices - 1) {
            indices.push(2 * numStacks - 2);
        } else {
            indices.push((2 * numStacks - 1) * (i + 1) + 2 * numStacks - 2);
        }
    }


    //indices = [0, 1, 5];
    return [vertices, indices, edges];
}


function initCylinder(baseRadius, topRadius, height, numSlices, numStacks, color, center, name) {
    var cylinderVertices = getCylinderVertices(baseRadius, topRadius, height, numSlices, numStacks, color, center);
    var cylinderInd = getCylinderIndex(numSlices, numStacks);
    var cylinderEdgeInd = getCylinderEdgeIndex(numSlices, numStacks);

    //console.log(cylinderEdgeInd);

    initVBO(name, cylinderVertices, cylinderInd, color);

}

function getCylinderVertices(baseRadius, topRadius, height, numSlices, numStacks, color, center) {
    // center is the center of the bottom of the Cylinder
    // color is a 3 element array

    var vertices = [];
    var start = [center[0] + baseRadius, center[1], center[2]];
    var rotCenter = [center[0], center[1], center[2]];
    var diffShift = (baseRadius - topRadius) / numStacks;
    var heightShift = height / numStacks;

    Array.prototype.push.apply(vertices, center);
    Array.prototype.push.apply(vertices, [0, -1, 0]); // Put norm just after each point

    for (var h = 0; h <= numStacks; ++h) {
        var diffDeg = 2 * Math.PI / numSlices;
        var rotMatrix = mat4.identity(mat4.create());
        rotMatrix = mat4.translate(rotMatrix, rotCenter);
        rotMatrix = mat4.rotateY(rotMatrix, diffDeg);
        rotMatrix = mat4.translate(rotMatrix, [-rotCenter[0], -rotCenter[1], -rotCenter[2]]);

        var point = [start[0], start[1], start[2]];
        for (var i = 0; i < numSlices; ++i) {
            Array.prototype.push.apply(vertices, point);
            Array.prototype.push.apply(vertices, [point[0] - rotCenter[0], point[1] - rotCenter[1], point[2] - rotCenter[2]]);
            point = mat4.multiplyVec3(rotMatrix, point);
        }
        start = [start[0] - diffShift, start[1] + heightShift, start[2]];
        rotCenter  = [rotCenter[0], rotCenter[1] + heightShift, rotCenter[2]];
    }

    Array.prototype.push.apply(vertices, [center[0], center[1] + height, center[2]]);
    Array.prototype.push.apply(vertices, [0, 1, 0]);

    // now 1 + numSlices * (numStacks + 1) + 1  vertices   1 + 20 * 2 + 1 =42
    for (var i = 1; i <= numSlices; ++i) {
        var point = [vertices[6 * i ], vertices[6 * i + 1], vertices[6 * i + 2]];
        Array.prototype.push.apply(vertices, point);
        Array.prototype.push.apply(vertices, [0, -1, 0]);
    }
    // now 1 + numSlices * (numStacks + 1) + 1 + numSlices vertices
    for (var i = 1; i <= numSlices; ++i) {
        var point = [vertices[6 * (i + numSlices * numStacks) ], vertices[6 * (i + numSlices * numStacks) + 1], vertices[6 * (i + numSlices * numStacks) + 2]];
        Array.prototype.push.apply(vertices, point);
        Array.prototype.push.apply(vertices, [0, 1, 0]);
    }

    //  numslices = 4
    // 0
    //  1 2 3 4
    //  5 6 7 8
    // 9
    // 10 11 12 13
    // 3 + 2 + 4 * 2
    // 14 15 16 17


    return vertices;
}

function getCylinderIndex(numSlices, numStacks) {
    var indices = []


    for (var i = 1; i < numSlices; ++i) {
        indices.push(0);
        indices.push(i + 1 + numSlices * (numStacks + 1));
        indices.push(i + 2 + numSlices * (numStacks + 1));
    }

    // numSlices - 1  + 2 + numSlices * (numStacks + 1)

    indices.push(0);
    indices.push(2 + numSlices * (numStacks + 1));
    indices.push(1 + numSlices * (numStacks + 2));

    for (var h = 0; h < numStacks; ++h) {
        for (var i = 1; i < numSlices; ++i) {
            indices.push(h * numSlices + i);
            indices.push(h * numSlices + i + 1);
            indices.push((h + 1) * numSlices + i);

            indices.push(h * numSlices + i + 1);
            indices.push((h + 1) * numSlices + i);
            indices.push((h + 1) * numSlices + i + 1);

        }

        indices.push((h + 1) * numSlices);
        indices.push((h + 2) * numSlices);
        indices.push(h * numSlices + 1);

        indices.push(h * numSlices + 1);
        indices.push((h + 2) * numSlices);
        indices.push((h + 1) * numSlices + 1);
    }


    for (var i = 1; i < numSlices; ++i) {
        indices.push((numStacks + 1) * numSlices + 1);
        indices.push(i +  numSlices * (numStacks + 2) + 1);
        indices.push(i + 1 + numSlices * (numStacks + 2) + 1);
    }

    indices.push((numStacks + 1) * numSlices + 1);
    indices.push(numSlices * (numStacks + 3) + 1);
    indices.push(1  + numSlices * (numStacks + 2) + 1);

    return indices;
}

function getCylinderEdgeIndex(numSlices, numStacks) {
    var indices = [];
    for (var i = 1; i < numSlices; ++i) {
        indices.push(i);
        indices.push(i + 1);
    }

    indices.push(numSlices);
    indices.push(1);

    for (var i = 1; i < numSlices; ++i) {
        indices.push(i + numSlices * numStacks);
        indices.push(i + numSlices * numStacks + 1);
    }
    indices.push(numSlices * (1 + numStacks));
    indices.push(1 + numSlices * numStacks);

    return indices;

}

function drawStar2d(short, long) {
    var point = [0, short, 0];
    var rotMat = mat4.rotateZ(mat4.identity(mat4.create()), 2* Math.PI/10);
    point = mat4.multiplyVec3(rotMat, point);

    var pointl = [0, long, 0];

    var rotMat = mat4.rotateZ(mat4.identity(mat4.create()), -2 * Math.PI/5);

    var vertices = [];
    for (var i = 0; i < 5; ++i) {
        vertices.push(point[0]);
        vertices.push(point[1]);

        vertices.push(pointl[0]);
        vertices.push(pointl[1]);

        point = mat4.multiplyVec3(rotMat, point);
        pointl = mat4.multiplyVec3(rotMat, pointl);
    }

    return vertices;
}

var surfaceIndices;
var surfaceIndBuffer;

// Parts of the below code is highly adapted from the 3Dcube.js in the given tutorial
function initBuffers() {
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "drawEdge"), 0);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "reverseNorm"), 0);
   // Color based on color-triangles.js

    var size = 8;
    var surfaceVertices = [size, -size, size,
                -size, -size, size,
               -size, -size,  -size,
                size, -size,  -size,
               ];

    var surfaceIndices = [0, 2, 1, 0, 3, 2, 1, 0, 2, 0, 3, 2];
    //0, 1, 2, 3, 0, 2];
    var edges = [0, 1, 1, 2, 2, 3, 3, 0];

    //var norms = getTriangleNorms(surfaceVertices, surfaceIndices, 3);
    var norms = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0];
    //initVBOwNorm("surface", surfaceVertices, 4, surfaceIndices, 12, norms, [.9, .9, .8]);

    // initVBOwNormATex(name, vertices, verticeNum, indices, indiceNum, norms,
    //                      diffuse_mat, specular_mat, texPic, texLoc)

    //var texLoc = [1, 1, 1, 0, 0, 0, 0, 1];
    var texLoc = [0, 1, 1, 1, 1, 0, 0, 0, ];
    initVBOwNormATex("surfacenegy", surfaceVertices, 4, surfaceIndices, 12, norms,
                     diffuse_mat, specular_mat, "static/images/negy.jpg", texLoc);

    var surfaceNegZVertices = [size, size, -size,
                               size, -size, -size,
                               -size, -size, -size,
                               -size, size, -size
                              ];
    var normsNegZ = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1];
    texLoc = [1, 1, 1, 0, 0, 0, 0, 1];
    initVBOwNormATex("surfacenegz", surfaceNegZVertices, 4, surfaceIndices, 12, normsNegZ,
                     diffuse_mat, specular_mat, "static/images/negz.jpg", texLoc);

    var surfacePosXVertices = [size, size, size,
                               size, size, -size,
                               size, -size, -size,
                               size, -size, size
                              ];
    var normsPosX = [-1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0];
    texLoc = [1, 1, 0, 1, 0, 0, 1, 0];
    initVBOwNormATex("surfaceposx", surfacePosXVertices, 4, surfaceIndices, 12, normsPosX,
                     diffuse_mat, specular_mat, "static/images/posx.jpg", texLoc);


    var surfaceNegXVertices = [-size, size, size,
                               -size, size, -size,
                               -size, -size, -size,
                               -size, -size, size
                              ];
    var normsNegX = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0];
    texLoc = [0, 1, 1, 1, 1, 0, 0, 0];
    initVBOwNormATex("surfacenegx", surfaceNegXVertices, 4, surfaceIndices, 12, normsNegX,
                     diffuse_mat, specular_mat, "static/images/negx.jpg", texLoc);

    var surfacePosZVertices = [size, size, size,
                               size, -size, size,
                               -size, -size, size,
                               -size, size, size
                              ];
    var normsPosZ = [-1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0];
    texLoc = [0, 1, 0, 0, 1, 0, 1, 1];
    initVBOwNormATex("surfaceposz", surfacePosZVertices, 4, surfaceIndices, 12, normsPosZ,
                     diffuse_mat, specular_mat, "static/images/posz.jpg", texLoc);

    var surfacePosYVertices = [size, size, size,
                               size, size, -size,
                               -size, size, -size,
                               -size, size, size
                              ];
    var normsPosY = [0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0];
    texLoc = [0, 0, 0, 1, 1, 1, 1, 0];
    initVBOwNormATex("surfaceposy", surfacePosYVertices, 4, surfaceIndices, 12, normsPosY,
                     diffuse_mat, specular_mat, "static/images/posy.jpg", texLoc);



    initCylinder(0.3, 0.3, 0.2, 20, 1, [0, 0.2, 1], [2, -0.2, -1.0], "cylinder1");
    initCylinder(0.5, 0.01, 0.9, 30, 3, [0.3, 1.0, 0.0], [2, -0.5, 2], "cylinder2");

    initSphere(0.1, 20, 10, [1.0, 0.5, 0.0], [-1.5, 1.8, 0], "sphere1");
    initSphere(0.3, 10, 10, [1, 0, 0], [2.3, 1.8, 1.5], "sphere2");

    initSphere(0.05, 10, 10, [1.0, 1.0, 0.0], [pos_light[0], pos_light[1], pos_light[2]], "sphere_light");

    initCube(1, [0, 1.0, 0], [0, 0, 0], "cube_ref");

    initSphere(0.5, 20, 20, [1.0, 1.0, 0.0], [0, 1, 0], "sphere_ref");

    initCube(0.2, [1.0, 0.0, 0.0], [-1, -.3, -2], "cube1");

    initCube(0.2, [1.0, 0.0, 0.0], [1.0, -0.3, 2.0], "cube2");

    // Point are listed in clockwise order
    initPoly([0.6, 0.4, 0.6, -0.4, -0.6, -0.4, -0.6, 0.4], 0.05, [0.5, 1, 0.9], [0, Math.PI/2, 0.0], [1.0, 1.0, 1.0], [-2.25, 1.6, 0], "poly1");
    initPoly([0.2, 0.01, .2, -0.01, -.2, -.01, -0.2, 0.01], 0.005, [1.0, 0, 0], [0, Math.PI/2, 0.0], [1.0, 1.0, 1.0], [-2.2, 1.31, 0], "poly2");

    initPoly([0.2, 0.01, .2, -0.01, -.2, -.01, -0.2, 0.01], 0.005, [1.0, 0, 0], [0, Math.PI/2, 0.0], [1.0, 1.0, 1.0], [-2.2, 1.51, 0], "poly3");

    initPoly([0.005, 0.1, .005, -0.1, -.005, -.1, -0.005, 0.1], 0.005, [1.0, 0, 0], [0, Math.PI/2, 0.0], [1.0, 1.0, 1.0], [-2.2, 1.41, 0.2], "poly4");
    initPoly([0.005, 0.1, .005, -0.1, -.005, -.1, -0.005, 0.1], 0.005, [1.0, 0, 0], [0, Math.PI/2, 0.0], [1.0, 1.0, 1.0], [-2.2, 1.41, -0.2], "poly5");

    initCylinder(0.15, 0.18, 0.3, 10, 3, [1.0, 0.0, 0.0], [-2, 1, 0.0], "cylinder3");

    // Colors are referenced by https://www.colorspire.com/rgb-color-wheel/
    initPoly(drawStar2d(0.15, 0.3), 0.1, [0.99, 0.99, 0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2, 0.7, 2], "polystar");


    initCube(0.2, [1.0, 0.0, 0.0], [-0.2, -0.3, 2.3], "cube10");
    initCube(0.2, [0.0, 1.0, 0.0], [0.2, -0.3, 2.3], "cube11");
    initCube(0.2, [0.0, 0.0, 1.0], [0, -0.3, 2.5], "cube12");
    initCube(0.2, [1.0, 0.0, 0.0], [-0.2, -0.3, 2.7], "cube13");
    initCube(0.2, [0.0, 1.0, 0.0], [0.2, -0.3, 2.7], "cube14");

    initCube(0.2, [0.0, 1.0, 1.0], [-0.2, -0.1, 2.5], "cube20");
    initCube(0.2, [1.0, 0.0, 1.0], [0.2, -0.1, 2.5], "cube21");
    initCube(0.2, [0.0, 0.0, 1.0], [0, -0.1, 2.3], "cube22");
    initCube(0.2, [0.0, 0.0, 1.0], [0, -0.1, 2.7], "cube23");

    initCube(0.2, [0.0, 1.0, 0.0], [-0.2, 0.1, 2.3], "cube30");
    initCube(0.2, [1.0, 0.0, 0.0], [0.2, 0.1, 2.3], "cube31");
    initCube(0.2, [0.0, 0.0, 1.0], [0, 0.1, 2.5], "cube32");
    initCube(0.2, [0.0, 1.0, 0.0], [-0.2, 0.1, 2.7], "cube33");
    initCube(0.2, [1.0, 0.0, 0.0], [0.2, 0.1, 2.7], "cube34");


    initCylinder(0.2, 0.15, 0.1, 20, 1, [0.0, 0.8, 0.5], [-0.21, 0.2, 2.5], "cylinder4");

    initCylinder(0.09, 0.02, 0.2, 20, 1, [0.7, 0.7, 0.6], [0.2, 0.2, 2.3], "cylinder5");
    initSphere(0.05, 10, 10, [0.7, 0.7, 0.6], [0.2, 0.4 + 0.05,2.3], "sphere3");


    initPoly(drawStar2d(0.05, 0.09), 0.01, [1.0, 1.0, 0.0], [0, Math.PI/4, Math.PI/6], [3.0, 1.0, 1.0], [-.2, 0.4, 2.5], "polystar1");

    //initPoly(drawStar2d(0.5, 0.6), 0.2, [0.99, 0, 0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.5, 0.7, 1], "polystarbump");


}



function getMouseLocation(event) {
    var mouseX = event.clientX - canvasleftoffset;
    var mouseY = event.clientY - canvastopoffset;
    mouseX = (mouseX - 0.5 * gl.viewportWidth) / (0.5 * gl.viewportWidth);
    mouseY = (0.5 * gl.viewportHeight - mouseY) / (0.5 * gl.viewportHeight);

    // Use multiple return based on https://www.javascripttutorial.net/javascript-return-multiple-values/
    // also use this syntax in later code
    return [mouseX, mouseY]
}

// Based on the naming in https://developer.mozilla.org/en-US/docs/Web/CSS/cursor
// Syntax error: ? https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Errors/Missing_formal_parameter
function matvecproduct(mat, vec) {
    if (mat.length % vec.length != 0) {
        console.log("Matvecproduct not correct!");
        return;
    }
    // Constructor based on https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/Array
    var row = mat.length / vec.length;
    var res = new Array(row);
    for (var i = 0; i < row; ++i) {
        res[i] = 0.0;
        for (var j = 0; j < vec.length; ++j) {
            res[i] += mat[j * vec.length + i] * vec[j];
        }
    }

    return res;
}


// Interact based on the movement of mouse
// based on uniform-color-xform.js

var vMatrix = mat4.create(); // view matrix
var mMatrix = mat4.create();  // model matrix
var pMatrix = mat4.create();  //projection matrix
var mvMatrix = mat4.create();
var tMatrix = mat4.create();
var Z_angle = 0;
var cameraY = 0;
var cameraP = 0;
var cameraR = 0;
var cameraAt = [0, 2, 5];
var negCameraAt = [0, -2, -5];

function mymatproduct(tMatrix, vertices) {
    var res = [];
    for (var i = 0; i < vertices.length; i += 3) {
        var myvec = new Float32Array(4);
        myvec[0] = vertices[i];
        myvec[1] = vertices[i + 1];
        myvec[2] = vertices[i + 2];
        myvec[3] = 1.0;
        curpoint = matvecproduct(tMatrix, myvec);
        res.push(curpoint[0]);
        res.push(curpoint[1]);
        res.push(curpoint[2]);
    }
    return res;
}

var pos_light = [2, 2, 3, 1];
var ambient_mat = [1, 1, 1, 1];
var ambient_light = [0.1, 0.1, 0.1, 1];

var diffuse_mat = [0, 1, 1, 1];
var diffuse_light = [1, 1, 1, 1];

var specular_mat = [.9, .9, .9, 1];
var specular_light = [1, 1, 1, 1];

var shineval = 50;


function drawScene() {
    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    mat4.perspective(60, 1.0, 0.1, 100, pMatrix);  // set up the projection matrix
    mat4.identity(vMatrix);

    vMatrix = mat4.lookAt(cameraAt, [0, 0, 0], [0, 1, 0], vMatrix);	// set up the view matrix, multiply into the modelview matrix



    vMatrix = mat4.translate(vMatrix, cameraAt);
    vMatrix = mat4.rotate(vMatrix, cameraY, [0, 1, 0]);
    vMatrix = mat4.rotate(vMatrix, cameraP, [1, 0, 0]);
    vMatrix = mat4.rotate(vMatrix, cameraR, [0, 0, 1]);
    vMatrix = mat4.translate(vMatrix, negCameraAt);


    mat4.identity(mMatrix);
    //console.log('Z angle = '+ Z_angle);
    mMatrix = mat4.rotate(mMatrix, 0, [0, 1, 1]);   // now set up the model matrix
    //console.log("mMatrix");
    //console.log(mMatrix);
    mat4.multiply(vMatrix, mMatrix, mvMatrix);
    //mat4.multiply(pMatrix, mvMatrix, tMatrix);
    // Based on 12-shading.js
    var nMatrix = mat4.create();
    mat4.identity(nMatrix);
    nMatrix = mat4.multiply(nMatrix, vMatrix);
    nMatrix = mat4.multiply(nMatrix, mMatrix);
    // Based on 12-shading.js
    nMatrix = mat4.inverse(nMatrix);
    nMatrix = mat4.transpose(nMatrix);

    // Based on cubeMappedTeapot.js in the given tutorial
    var v2wMatrix = mat4.create();
    v2wMatrix = mat4.identity(v2wMatrix);
    v2wMatrix = mat4.multiply(v2wMatrix, vMatrix);
    v2wMatrix = mat4.transpose(v2wMatrix);


    var umMatrix = mat4.create();
    mat4.identity(umMatrix);
    umMatrix = mat4.multiply(umMatrix, mMatrix);
    umMatrix = mat4.inverse(umMatrix);
    umMatrix = mat4.transpose(umMatrix);

    //resvertices = mymatproduct(tMatrix, vertices);
    gl.uniformMatrix4fv(gl.getUniformLocation(shaderProgram, "uv2wMatrix"), false, v2wMatrix);
    gl.uniformMatrix4fv(shaderProgram.pMatrixLoc, false, pMatrix);
    gl.uniformMatrix4fv(shaderProgram.mMatLoc, false, mMatrix);
    gl.uniformMatrix4fv(shaderProgram.vMatLoc, false, vMatrix);
    gl.uniformMatrix4fv(gl.getUniformLocation(shaderProgram, "unMatrix"), false, nMatrix);
    gl.uniformMatrix4fv(gl.getUniformLocation(shaderProgram, "umMatrix"), false, umMatrix);
    // camera_pos is for cube map just
    gl.uniform3f(gl.getUniformLocation(shaderProgram, "camera_pos"), false, cameraAt[0], cameraAt[1], cameraAt[2]);


    //gl.bindBuffer(gl.ARRAY_BUFFER, surfaceVertexBuffer);
    //gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, 3, gl.FLOAT, false, 24, 0);
    //gl.vertexAttribPointer(shaderProgram.vertexColorAttribute, 3, gl.FLOAT, false, 24, 12);

    //gl.drawArrays(gl.LINE_LOOP, 0, 4);


    gl.uniform3f(gl.getUniformLocation(shaderProgram, "light_loc"), pos_light[0], pos_light[1], pos_light[2]);
    gl.uniform4f(gl.getUniformLocation(shaderProgram, "ambient_mat"), ambient_mat[0], ambient_mat[1], ambient_mat[2], 1.0);
    gl.uniform4f(gl.getUniformLocation(shaderProgram, "ambient_light"), ambient_light[0], ambient_light[1], ambient_light[2], 1.0);

    gl.uniform4f(gl.getUniformLocation(shaderProgram, "diffuse_mat"), diffuse_mat[0], diffuse_mat[1], diffuse_mat[2], 1.0);
    gl.uniform4f(gl.getUniformLocation(shaderProgram, "diffuse_light"), diffuse_light[0], diffuse_light[1], diffuse_light[2], 1.0);


    gl.uniform4f(gl.getUniformLocation(shaderProgram, "specular_light"), specular_light[0], specular_light[1], specular_light[2], 1.0);
    gl.uniform4f(gl.getUniformLocation(shaderProgram, "specular_mat"), specular_mat[0], specular_mat[1], specular_mat[2], 1.0);

    gl.uniform1f(gl.getUniformLocation(shaderProgram, "shine_val"), shineval);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "udrawTexture"), 0);


    //drawShapewNorm("surface", gl.TRIANGLES, 0);
    //drawShapewBump("polystarbump", gl.TRIANGLES, false);


    //drawShapeReflectTexture("cube_ref", true);

    drawShapewBump("cube_ref");


    drawShapewNorm("cube1", gl.TRIANGLES, 0);

    drawShape("sphere_light", gl.TRIANGLES, true);
    drawShape("cylinder1", gl.TRIANGLES, false);



    drawShape("cylinder2", gl.TRIANGLES, false);
    drawShape("sphere1", gl.TRIANGLES, false);




    drawShapewNorm("cube2", gl.TRIANGLES, 0);

    drawShape("poly1", gl.TRIANGLES, false);
    drawShape("poly2", gl.TRIANGLES, false);
    drawShape("poly3", gl.TRIANGLES, false);
    drawShape("poly4", gl.TRIANGLES, false);
    drawShape("poly5", gl.TRIANGLES, false);

    drawShape("cylinder3", gl.LINES, false);

    drawShape("sphere2", gl.LINES, false);

    drawShape("polystar", gl.TRIANGLES, false);


    drawShapewNorm("cube11", gl.TRIANGLES, 0);
    drawShapewNorm("cube12", gl.TRIANGLES, 0);
    drawShapewNorm("cube10", gl.TRIANGLES, 0);
    drawShapewNorm("cube13", gl.TRIANGLES, 0);
    drawShapewNorm("cube14", gl.TRIANGLES, 0);

    drawShapewNorm("cube20", gl.TRIANGLES, 0);
    drawShapewNorm("cube21", gl.TRIANGLES, 0);
    drawShapewNorm("cube22", gl.TRIANGLES, 0);
    drawShapewNorm("cube23", gl.TRIANGLES, 0);

    drawShapewNorm("cube30", gl.TRIANGLES, 0);
    drawShapewNorm("cube31", gl.TRIANGLES, 0);
    drawShapewNorm("cube32", gl.TRIANGLES, 0);
    drawShapewNorm("cube33", gl.TRIANGLES, 0);
    drawShapewNorm("cube34", gl.TRIANGLES, 0);


    drawShape("cylinder4", gl.TRIANGLES, false);
    drawShape("cylinder5", gl.TRIANGLES, false);

    drawShape("polystar1", gl.TRIANGLES, false);

    drawShape("sphere3", gl.TRIANGLES, false);



    drawShapeReflectTextureCombNorm("sphere_ref");
    drawShapeReflectTextureCombNorm("sphere_ref");


    //drawShapeReflectTextureNoBump("cube_ref");


    drawShapewTexture("surfacenegy", 2);
    drawShapewTexture("surfacenegz", 2);
    drawShapewTexture("surfaceposx", 2);
    drawShapewTexture("surfacenegx", 2);
    drawShapewTexture("surfaceposz", 2);
    drawShapewTexture("surfaceposy", 2);


    //drawShapewBump("polystarbump", gl.TRIANGLES, false);


    if (jsonready == 1) {
        drawShapewTexture("car", 1);
    }



}


function twoNorm(x, y) {
    return (x[0] - y[0]) ** 2 + (x[1] - y[1])**2;
}

// Check points inside a polygon
// based on https://stackoverflow.com/questions/1119627/how-to-test-if-a-point-is-inside-of-a-convex-polygon-in-2d-integer-coordinates
// Check if a point is in the same side https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
function pointInSameSide(point, poly) {
    if (poly.length < 4) {
        console.log("The polygon should have at least three points");
        return false;
    }
    if (poly[poly.length - 1] != poly[0]) {
        console.log("The last and the first point should be the same for a valid polygon");
        return false;
    }
    var sign = false;
    for (var i = 1; i < poly.length; ++i) {
        p1 = poly[i - 1];
        p2 = poly[i];
        var nowsign = ((point[0] - p1[0]) * (p2[1] - p1[1]) - (point[1] - p1[1]) * (p2[0] - p1[0])) > 0 ? true : false;
        if (i == 1) {
            sign = nowsign;
            continue;
        } else if (nowsign != sign) {
            return false;
        }
    }
    return true;
}

var vboMap = new Map();
var actionMap = new Map();

// Use map based on https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map

function moveObj(objectname, action, step) {
    var vbo = vboMap.get(objectname);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);

    vertices = vbo.vertices;
    var trMatrix = mat4.identity(mat4.create());
    trMatrix = mat4.translate(trMatrix, action);

    applyVertices(trMatrix, vertices, step);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
}

function moveObjwCenter(objectname, action, center, step) {
    var vbo = vboMap.get(objectname);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);

    vertices = vbo.vertices;
    var cenMatrix = mat4.identity(mat4.create());
    cenMatrix = mat4.translate(cenMatrix, [-center[0], -center[1], -center[2]]);

    var trMatrix = mat4.identity(mat4.create());
    trMatrix = mat4.translate(trMatrix, action);

    var cenMatrix2 = mat4.identity(mat4.create());
    cenMatrix2 = mat4.translate(cenMatrix2, [center[0], center[1], center[2]]);

    applyVertices(cenMatrix, vertices, step);
    applyVertices(trMatrix, vertices, step);
    applyVertices(cenMatrix2, vertices, step);

    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
}

function rotateObjwCenter(objectname, deg, center, step) {
    var vbo = vboMap.get(objectname);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);

    vertices = vbo.vertices;

    var cenMatrix = mat4.identity(mat4.create());
    cenMatrix = mat4.translate(cenMatrix, [-center[0], -center[1], -center[2]]);

    var rotMatrix = mat4.identity(mat4.create());
    rotMatrix = mat4.rotateY(rotMatrix, deg);

    var cenMatrix2 = mat4.identity(mat4.create());
    cenMatrix2 = mat4.translate(cenMatrix2, [center[0], center[1], center[2]]);


    applyVertices(cenMatrix, vertices, step);
    applyVertices(rotMatrix, vertices, step);
    applyVertices(cenMatrix2, vertices, step);

    if (vboMap.has(objectname + "_norm") == false) {
        applyVerticeswStart(rotMatrix, vertices, step, 3);

        vbo.vertices = vertices;
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
        return;
    }

    vbo.vertices = vertices;
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);



    var vbo_norm = vboMap.get(objectname + "_norm");
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo_norm);

    norms = vbo_norm.norms;
    var rotMatrix = mat4.identity(mat4.create());
    rotMatrix = mat4.rotateY(rotMatrix, deg);

    //applyVertices(cenMatrix, norms, step);
    applyVertices(rotMatrix, norms, step);
    //applyVertices(cenMatrix2, norms, step);

    vbo.norms = norms;
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(norms), gl.STATIC_DRAW);
}

function applyVertices(trMatrix, vertices, step) {
    for (var i = 0; i < vertices.length; i += step) {
        var vec = [vertices[i], vertices[i + 1], vertices[i + 2]];
        var resVec = mat4.multiplyVec3(trMatrix, vec);
        vertices[i] = resVec[0];
        vertices[i + 1] = resVec[1];
        vertices[i + 2] = resVec[2];
    }

}

function applyVerticeswStart(trMatrix, vertices, step, start) {
    for (var i = start; i < vertices.length; i += step) {
        var vec = [vertices[i], vertices[i + 1], vertices[i + 2]];
        var resVec = mat4.multiplyVec3(trMatrix, vec);
        vertices[i] = resVec[0];
        vertices[i + 1] = resVec[1];
        vertices[i + 2] = resVec[2];
    }

}

var cameraAct = new Map();
cameraAct.set("B", [0.1, 0.0, 0.0]);
cameraAct.set("b", [-0.1, 0.0, 0.0]);
cameraAct.set("N", [0.0, 0.1, 0.0]);
cameraAct.set("n", [0.0, -0.1, 0.0]);
cameraAct.set("M", [0.0, 0.0, 0.1]);
cameraAct.set("m", [0.0, 0.0, -0.1]);

var lightAct = new Map();
lightAct.set("J", [0.1, 0.0, 0.0]);
lightAct.set("j", [-0.1, 0.0, 0.0]);
lightAct.set("K", [0.0, 0.1, 0.0]);
lightAct.set("k", [0.0, -0.1, 0.0]);
lightAct.set("L", [0.0, 0.0, 0.1]);
lightAct.set("l", [0.0, 0.0, -0.1]);


var curMoveInd = 0;
var moveObjList = [["sphere_ref", 3], ["cube1", 3], ["cube2", 3], ["cylinder1", 6], ["cylinder2", 6,  "polystar", 6],
                   ["sphere1", 6], ["sphere2", 6], ["poly1", 6, "poly2", 6, "poly3", 6, "poly4", 6, "poly5", 6, "cylinder3", 6],
                   ["cube10", 3, "cube11", 3, "cube12", 3, "cube13", 3, "cube14", 3,
                    "cube20", 3, "cube21", 3, "cube22", 3, "cube23", 3,
                    "cube30", 3, "cube31", 3, "cube32", 3, "cube33", 3, "cube34", 3,
                    "cylinder4", 6, "polystar1", 6, "cylinder5", 6, "sphere3", 6],
                    ["cylinder4", 6, "polystar1", 6],
                    ["cylinder5", 6, "sphere3", 6],
                    ["cylinder4", 6, "polystar1", 6, "cylinder5", 6, "sphere3", 6],
                    ["polystar1", 6],
                    ["car", 3]
                  ];


// Check if string contains substr
// use includes
// based on https://sentry.io/answers/string-contains-substring-javascript/
function onDocumentKeyDown(event) {
    //console.log(event);

    if (event.key == "P") {
        cameraP += 0.01;
    } else if (event.key == "p") {
        cameraP -= 0.01;
    } else if (event.key == "Y") {
        cameraY += 0.01;
    } else if (event.key == "y") {
        cameraY -= 0.01;
    } else if (event.key == "R") {
        cameraR += 0.01;
    } else if (event.key == "r") {
        cameraR -= 0.01;
    } else if (event.key == "A" || event.key == "D" || event.key == "W" || event.key == "S" || event.key == "U" || event.key == "u") {
        var off = actionMap.get(event.key);
        for (var i = 0; i < moveObjList[curMoveInd].length; i += 2) {
            moveObj(moveObjList[curMoveInd][i], off, moveObjList[curMoveInd][i + 1]);
        }
    } else if (event.key == "B" || event.key == "b" ||
               event.key == "N" || event.key == "n" ||
               event.key == "M" || event.key == "m") {
        var off = cameraAct.get(event.key);
        cameraAt =  [cameraAt[0] + off[0], cameraAt[1] + off[1], cameraAt[2] + off[2]];
        negCameraAt =  [negCameraAt[0] - off[0], negCameraAt[1] - off[1], negCameraAt[2] - off[2]];

    } else if (event.key == " ") {
        curMoveInd += 1;
        if (curMoveInd == moveObjList.length) {
            curMoveInd = 0;
        }
    } else if (event.key == "J" || event.key == "j" ||
               event.key == "K" || event.key == "k" ||
               event.key == "L" || event.key == "l") {
        //changeBasedOnMap(event.key, pos_light, lightAct, 1);
        console.log(pos_light);
        var off = lightAct.get(event.key);
        pos_light = [pos_light[0] + off[0], pos_light[1] + off[1], pos_light[2] + off[2]];
        moveObj("sphere_light", off, 6);
    }

    drawScene();
}

// Add animation based on
// https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Animating_objects_with_WebGL
let then = 0.0;
var camRotDegree = 0.0;
var rightcnt = 0.0;
var dir = 1.0;

function update(now) {
    now *= 0.001;
    var deltaTime = now - then;
    then = now;
    camRotDegree += deltaTime;

    var rotMatY = mat4.identity(mat4.create());
    rotMatY = mat4.rotateY(rotMatY, 0.001);
    applyVertices(rotMatY, cameraAt, 3);


    negCameraAt[0] = -cameraAt[0];
    negCameraAt[1] = -cameraAt[1];
    negCameraAt[2] = -cameraAt[2];

    var off = [0.01 * Math.sin(now), 0.0, 0.0];
    //console.log(off);

    moveObjwCenter("sphere_ref", off, [0, 2, 0], 3);
    if (jsonready == 1) {
        rotateObjwCenter("car", -0.005, [-2, -0.1, 2.0], 3);
    }
    rotateObjwCenter("cube2", 0.005, [1, -0.3, 2], 3);
    rotateObjwCenter("polystar", -0.01, [2, 0.7, 2.05], 6);
    // moveObj("sphere_ref", off, 3);


    drawScene();

    requestAnimationFrame(update);

}

function webGLStart() {
    var canvas = document.getElementById("hw-canvas");
    // Offset based on https://www.geeksforgeeks.org/how-to-find-the-position-of-html-elements-in-javascript/
    canvastopoffset = canvas.offsetTop;
    canvasleftoffset = canvas.offsetLeft;
    initGL(canvas);
    initShaders();

    gl.enable(gl.DEPTH_TEST);
    //gl.enable(gl.CULL_FACE);
    gl.clearColor(1, 1, 1, 1.0);
    // gl.clearColor(1, 1, 1, 1.0);

    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    initFromJson("car.json");

    initBuffers();

    gl.uniform1i(gl.getUniformLocation(shaderProgram, "udrawTexture"), 0);
    gl.uniform1i(gl.getUniformLocation(shaderProgram, "use2DTex"), 0);
    // Interactive based on uniform-color-xforms.js
    // What is false? see https://developer.mozilla.org/en-US/docs/Web/API/EventTarget/addEventListener

    // How to disable a button
    // based on https://stackoverflow.com/questions/34136053/how-to-grey-out-and-disable-a-button-after-its-been-pressed-once
    // Hide element based on https://www.w3schools.com/howto/howto_js_toggle_hide_show.asp


    actionMap.set("W", [0, 0, 0.1]);
    actionMap.set("S", [0, 0, -0.1]);
    actionMap.set("A", [0.1, 0, 0]);
    actionMap.set("D", [-0.1, 0, 0]);
    actionMap.set("U", [0.0, 0.1, 0.0]);
    actionMap.set("u", [0.0, -0.1, 0.0]);

    document.addEventListener("keydown", onDocumentKeyDown, false);

    // Set timeout based on https://developer.mozilla.org/en-US/docs/Web/API/setTimeout


    initCubeMapTexture();
    drawScene();
    //
    requestAnimationFrame(update);

}



//References for the code
//  1. Following files in the https://github.com/hguo/WebGL-tutorial.git
//     open-canvas.js
//     3Dcube.js
//     simple-triangles.js
//     8-transform-ortho2D.js
//     3Dcube.js
//     solor.html
//     12-shading.js and 12-shading.js
//     cubeMappedTeapot.html and cubeMappedTeapot.js
//  2. The MDN web docs are also very helpful for this code https://developer.mozilla.org/en-US/
//  3. The stackoverflow and stackexchange discussion about how to check if a point is inside a polygon is very helpful for this homework
//  4. https://stackoverflow.com/questions/1119627/how-to-test-if-a-point-is-inside-of-a-convex-polygon-in-2d-integer-coordinates
//  5. https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
//  6. The cursor is based on https://developer.mozilla.org/en-US/docs/Web/CSS/cursor
//  7. The tutorial about camera https://learnwebgl.brown37.net/07_cameras/camera_rotating_motion.html
//  8. The bump mapping at the websites https://math.hws.edu/graphicsbook/source/webgl/bumpmap.html and https://apoorvaj.io/exploring-bump-mapping-with-webgl/
//  9. Some other references https://community.khronos.org/t/glsl-if-else-if-what-am-i-doing-wrong/66166
// TODO merge the cube vertice and color vbo together?
