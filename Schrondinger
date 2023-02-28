function Schrodinger(gpgpUtility_, xResolution_, length_, dt_)
{
  "use strict";

  var dt;
  var dtHandle;
  var fbos;
  /** WebGLRenderingContext */
  var gl;
  var gpgpUtility;
  var length;
  var lengthHandle;
  var pixels;
  var positionHandle;
  var potential;
  var potentialHandle;
  var program;
  var step;
  var textureCoordHandle;
  var textures;
  var waveFunctionHandle;
  var xResolution;
  var xResolutionHandle;

 
  this.createProgram = function (gl)
  {
    var fragmentShaderSource;
    var program;

 
    program            = gpgpUtility.createProgram(null, fragmentShaderSource);
    positionHandle     = gpgpUtility.getAttribLocation(program,  "position");
    gl.enableVertexAttribArray(positionHandle);
    textureCoordHandle = gpgpUtility.getAttribLocation(program,  "textureCoord");
    gl.enableVertexAttribArray(textureCoordHandle);
    dtHandle           = gl.getUniformLocation(program, "dt");
    potentialHandle    = gl.getUniformLocation(program, "potential");
    waveFunctionHandle = gl.getUniformLocation(program, "waveFunction");
    xResolutionHandle  = gl.getUniformLocation(program, "xResolution");
    lengthHandle       = gl.getUniformLocation(program, "length");

    return program;
  };

  /**
   * Setup the initial values for textures. 
   */
  this.setInitialTextures = function(texture0, texture1)
  {
    textures[0] = texture0;
    fbos[0]     = gpgpUtility.attachFrameBuffer(texture0);
    textures[1] = texture1;
    fbos[1]     = gpgpUtility.attachFrameBuffer(texture1);
  }

  /**
   * Set the potential as a texture
   */
  this.setPotential = function(texture)
  {
    potential = texture;
  }

  /**
   * Runs the program to do the actual work. On exit the framebuffer &amp;
   * texture are populated with the next timestep of the wave function.
   * You can use gl.readPixels to retrieve texture values.
   */
  this.timestep = function()
  {
    var gl;

    gl = gpgpUtility.getComputeContext();

    gl.useProgram(program);

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbos[step]);

    // Step switches back and forth between 0 and 1,
    // ping ponging the source & destination textures.
    step = (step+1)%2;

    gpgpUtility.getStandardVertices();

    gl.vertexAttribPointer(positionHandle,     3, gl.FLOAT, gl.FALSE, 20, 0);
    gl.vertexAttribPointer(textureCoordHandle, 2, gl.FLOAT, gl.FALSE, 20, 12);

    gl.uniform1f(dtHandle,          dt);
    gl.uniform1i(xResolutionHandle, xResolution);
    gl.uniform1f(lengthHandle,      length);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, textures[step]);
    gl.uniform1i(waveFunctionHandle, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, potential);
    gl.uniform1i(potentialHandle, 1);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  /**
   * Retrieve the most recently rendered to texture.
   *
   * @returns {WebGLTexture} The texture used as the rendering target in the most recent
   *                         timestep.
   */
  this.getRenderedTexture = function()
  {
      return textures[(step+1)%2];
  }

  /**
   * Retrieve the frambuffer that wraps the texture to be used as the source in the
   * next timestep. Render to this FBO in the initialization step.
   *
   * @returns {WebGLFramebuffer} The framebuffer wrapping the source texture for the next timestep.
   */
  this.getSourceFramebuffer = function()
  {
    return fbos[(step+1)%2];
  }

  /**
   * Invoke to clean up resources specific to this program. We leave the texture
   * and frame buffer intact as they are used in followon calculations.
   */
  this.done = function ()
  {
    gl.deleteProgram(program);
  };

  dt          = dt_;
  gpgpUtility = gpgpUtility_;
  gl          = gpgpUtility.getGLContext();
  program     = this.createProgram(gl);
  fbos        = new Array(2);
  length      = length_;
  textures    = new Array(2);
  step        = 0;
  xResolution = xResolution_;
};
