This homework follows the instructions given on Lab 5 Carmen. Based on the instructions, to run this code please run python -m http.server in this directory and then open http://0.0.0.0:8000/hw5.html

The environment cube mappping images are from https://www.humus.name/index.php?page=Textures&ID=31

The bump mapping code is based from the tutorial at https://apoorvaj.io/exploring-bump-mapping-with-webgl/ Bump mapping images are downloaded from https://apoorvaj.io/ which are from https://learnopengl.com/Advanced-Lighting/Parallax-Mapping I also check the website at https://math.hws.edu/graphicsbook/source/webgl/bumpmap.html


Knowing issues in this homework

  1. The Reflection of the cubemap has some issues that sometimes the image is upside down. Sometimes the reflection turns to black. The reflection seems not to be correct. It indicates that there are bugs in this environment reflection but I fail to find a way to fix that. To work around the upside down images issue, I actually edit the images for the reflection of the cube mapping.

  2. The normal direction used in the bump mapping does not seem to be correct, as the shadow does not look as expected when the lignt is moving. It only implements the normal mapping currectly.

  3. Based on the https://gamedev.stackexchange.com/questions/32543/glsl-if-else-statement-unexpected-behaviour and https://www.khronos.org/opengl/wiki/Sampler_(GLSL)#Texture_lookup_in_shader_stages Even when there are no shader compilation error, improper shader code may still cause undefined behaviors. This code is not well organized which may lead to some unexpected behaviors in the shader code and cause bugs.

  4. Some issues when moving the camera using Y/y, P/p, R,r. For example, if reducing the pitch angle, the camera does not move perfectly vertically.


References for this lab homework
  1. Following files in the https://github.com/hguo/WebGL-tutorial.git
     open-canvas.js
     3Dcube.js
     simple-triangles.js
     8-transform-ortho2D.js
     3Dcube.js
     solor.html
     12-shading.js and 12-shading.js
     cubeMappedTeapot.html and cubeMappedTeapot.js
  2. The MDN web docs are helpful to this homework https://developer.mozilla.org/en-US/ Some detailed links are commented in the js code
  3. The stackoverflow and stackexchange discussion about how to check if a point is inside a polygon is helpful for this homework. The links are https://stackoverflow.com/questions/1119627/how-to-test-if-a-point-is-inside-of-a-convex-polygon-in-2d-integer-coordinates  https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
  4. The tutorial about camera https://learnwebgl.brown37.net/07_cameras/camera_rotating_motion.html
  5. The bump mapping at websites https://apoorvaj.io/exploring-bump-mapping-with-webgl/ and https://math.hws.edu/graphicsbook/source/webgl/bumpmap.html
  6. The in-class demos of other students in the class are helpful to this homework.
