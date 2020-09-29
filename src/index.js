import "./style.css";
import * as THREE from "./js/three/three.module.js";
import { EffectComposer } from "./js/three/EffectComposer.js";
import { RenderPass } from "./js/three/RenderPass.js";

{
  let samples = 1;
  let canvas = document.querySelector('.c');

  document.querySelector('.samples__select').addEventListener('change', e => {
    samples = e.currentTarget.value;
    const body = document.querySelector('body');
    body.removeChild(canvas);
    canvas = document.createElement('canvas');
    canvas.classList.add('c');
    body.appendChild(canvas);
    main();
  });

  const fragmentShader = `
  #include <common>

  uniform vec3 iResolution;
  uniform float iTime;
  uniform sampler2D iChannel0;
  uniform vec3 vw_dir;
  uniform vec3 vw_pos;
  uniform mat3 camera_matrix;
  uniform int samples;


  #define M_PI 3.1415926535897932384626433832795
  const float PHI = 1.61803398874989484820459;
  const float infinity = 10000000000000000000.0;
  vec3 view_dir = vec3(-0.1, -0.1, -1.0);
  vec3 view_pos = vec3(0.0, 4.0, 25.0);

  float gold_noise(in vec2 xy, in float _seed)
  {
    return fract(tan(distance(xy*PHI, xy)*_seed++)*(xy.x)*cos(_seed*iTime));
  }

  const int MAX_DEPTH = 3;
  const int NUM_OF_OBJECTS = 6;
  const int NUM_OF_LIGHTS = 2;
  const float GAMMA = 2.2;


  const float bias = 0.000001;
  const float minimum_collision_distance = 0.00001;
  const float fov = 60.0 * M_PI / 180.0;

  const float EV = 10.0;
  const float light_intensity_scale = 10.0;
  const float light_directionality = 0.75;
  const float alpha_for_diffCosW = 1000.0;

  struct Ray{
    vec3 m_dir;
    vec3 m_or;
    vec3 energy;
    float ior;
  };

  struct Material{
  vec3 color;
    float ambient;
    float spec;
    float emissivity;
  };

  struct Object{
    int type;
    vec3 pos;
    float radius;
    vec3 normal;

    Material mat;

  };

  float seed = 0.0;
  vec2 uv = vec2(0.0);

  float random() {
  return gold_noise(uv, seed++);
  }

  mat3 make_rot_matrix_from_normals(vec3 _u,vec3 _v){
    float c = dot(_u, _v);
    mat3 _R = mat3(-1.0);
    if (c == 1.0){
        return mat3(1.0);
    }
    if (c != 1.0){
        vec3 _a = cross(_u, _v);
        mat3 _a_cross = mat3(0.0);
        _a_cross[1][0] = -_a.z;
        _a_cross[2][0] = _a.y;
        _a_cross[0][1] = _a.z;
        _a_cross[2][1] = -_a.x;
        _a_cross[0][2] = -_a.y;
        _a_cross[1][2] = _a.x;

        mat3 a_cross_sq = _a_cross * _a_cross;

        _R = mat3(1.0) + _a_cross + a_cross_sq * (1.0 / (1.0 + c));

    }
    return _R;
  }

  vec3 diff_cos_weighted_direction(float _spec, vec3 _n){
    if (_spec == 1.0){
        return _n;
    }

    float m = pow(alpha_for_diffCosW, _spec * _spec);

    float _u = random();
    float _v = random();

    float cosT = pow(_u, 1.0 / (1.0 + m));
    float sinT = sqrt(1.0 - cosT * cosT);
    float phi = 2.0 * M_PI * _v;

    vec3 step1 = vec3(sinT * cos(phi), sinT * sin(phi), cosT);


    vec3 rotated_vec = make_rot_matrix_from_normals(vec3(0.0, 0.0, 1.0), _n) * step1;

    return normalize(rotated_vec);


  }

  Ray create_new_ray(vec3 _dir, vec3 _pos){
    return Ray(_dir, _pos, vec3(1.0),1.0);


  }

  struct Light{
        vec3 pos;
        vec3 color;
  };


  Object scene[NUM_OF_OBJECTS];
  Object lights[NUM_OF_LIGHTS];

  vec3 interpolated_ray_dir(vec2 _xy, float w, float h){
    float aspect_ratio = w/h;
  float x = (2.0 * (_xy.x + 0.5) / w -1.0) *aspect_ratio * tan(fov * 0.5);
  float y = (1.0 - 2.0*(_xy.y + 0.5) / h) * (-1.0) * tan(fov/2.0);

    return normalize(camera_matrix*(vec3(x,y, -1.0)));
  }

  vec3 get_point_at_distance(Ray _ray, float d){
    return _ray.m_or + d * _ray.m_dir;
  }


  float intersect(Ray _ray, Object _obj){
    float d = infinity;
    if (_obj.type == 0 )
    {
          float d_t = dot(_obj.pos - _ray.m_or,_obj.normal)/dot(_obj.normal,_ray.m_dir);
        float ip_to_midpt = length(get_point_at_distance(_ray,d_t) - _obj.pos);
        if (ip_to_midpt < _obj.radius)
        {
            return d = d_t;
        }
    }

    if (_obj.type == 1){
          vec3 r_t_c = _ray.m_or - _obj.pos;
      float p1 = -dot(_ray.m_dir, r_t_c);
      float p2sqr = p1 * p1 - dot(r_t_c, r_t_c) + _obj.radius * _obj.radius;
        if (p2sqr < 0.0){
          return infinity;
        }
      float p2 = sqrt(p2sqr);
      float _d = min(p1 - p2, p1 + p2);
        return _d > 0.0? _d : infinity;
    }
    if (_obj.type == 2){
        float _N = dot(_obj.normal, _ray.m_dir);
        float _d = infinity;
        if (abs(_N) < bias){
            return infinity;
        }
        else{
            vec3 or_to_plane = _obj.pos - _ray.m_or;
          _d = dot(or_to_plane,_obj.normal) / _N;
        }
        if (_d > 0.0){
            return _d;
        }
        else{
            return infinity;
        }
    }

    return d;
  }


  vec3 get_normal(vec3 _point, Object _obj){
    vec3 n = vec3(0.0, 0.0, 1.0);

    if (_obj.type == 0){
        return normalize(_obj.normal);
    }
    if (_obj.type == 1){
        return normalize(_point - _obj.pos);
    }
    if (_obj.type == 2){
        return _obj.normal;
    }
    return n;
  }


  vec2 intersect_scene(inout Ray _ray, Object [NUM_OF_OBJECTS] _scene){
    float closest = infinity;
    int nearest_obj = -1;
    for(int i = 0; i< NUM_OF_OBJECTS; i++){
        float dist = intersect(_ray, _scene[i]);
        if (dist < closest){
            if(dist > minimum_collision_distance){
                closest = dist;
                nearest_obj = i;
            }
        }
    }
    return vec2(float(nearest_obj), closest);
  }

  vec3 trace(inout Ray _ray, Object [NUM_OF_OBJECTS] _scene, int _depth){
    vec3 acquired_color = vec3(0.0, 0.0, 0.0);
    if (_depth <= MAX_DEPTH){
        vec2 intersection_info = intersect_scene(_ray, _scene);
        int intersection_id = int(intersection_info.x);
        float intersection_distance = intersection_info.y;
        if (intersection_distance < infinity){
            float distance_sq = intersection_distance * intersection_distance;
            vec3 intersection_point = get_point_at_distance(_ray, intersection_distance);
            vec3 surf_norm = get_normal(intersection_point,_scene[intersection_id]);
            float n_dot_r = dot(surf_norm, _ray.m_dir);


            if (_scene[intersection_id].mat.emissivity > 0.0){
                acquired_color += _scene[intersection_id].mat.emissivity * _scene[intersection_id].mat.color * abs(1.0) ;

            }

            vec3 diffuse_direct_col = vec3(0.0);
            for(int k =0;k<1;k++){
                for(int i=0;i<NUM_OF_OBJECTS;i++){
                    if(_scene[i].mat.emissivity > 0.0){
                        vec3 pt_to_light_vec0 = normalize(_scene[i].pos - intersection_point);

                        vec3 random_dir = diff_cos_weighted_direction(light_directionality, pt_to_light_vec0);

                        Ray r_to_light = create_new_ray(random_dir, intersection_point + bias * surf_norm);

                        vec2 towards_light_intersection_info = intersect_scene(r_to_light, _scene);
                        int index_collision = int(towards_light_intersection_info.x);

                        if(towards_light_intersection_info.y < infinity){
                            if(_scene[index_collision].mat.emissivity > 0.0){
                                float n_dot_li = dot(random_dir, surf_norm);
                                if(n_dot_li > 0.0){
                                    float prob_weight = 0.5 - cos(pow(light_directionality,4.0)*M_PI)*0.5;
                                    float dist_to_light = length(_scene[i].pos - intersection_point);
                                    float di_fo = 1.0 / (dist_to_light*dist_to_light) * (prob_weight) + (1.0 - prob_weight);
                                    diffuse_direct_col +=  n_dot_li * di_fo * _scene[index_collision].mat.color *  _scene[index_collision].mat.emissivity;
                                }
                            }
                        }
                    }
                }
            }

            acquired_color += diffuse_direct_col * _scene[intersection_id].mat.color * (1.0 -_scene[intersection_id].mat.spec) ;

            float dice = random();
            _ray.m_or = intersection_point + bias * surf_norm;
            _ray.m_dir = diff_cos_weighted_direction( _scene[intersection_id].mat.spec, reflect(_ray.m_dir, surf_norm));
            _ray.energy *=  _scene[intersection_id].mat.color;


        }
        else{
            _ray.energy = vec3(0.0);
            return vec3(0.0);
        }
    }

    return acquired_color;
  }

  void make_scene(){
    float default_ambient = 0.000;
    float freq = 2.0;
    float glow = 0.5*cos(freq*iTime)+0.5;
    Material teal = Material(vec3(0.0, 1.0, 1.0), default_ambient, 0.5, default_ambient);
    Material gold = Material(vec3(0.83, 0.68, 0.216), default_ambient, 1.0, default_ambient);
    Material white = Material(vec3(1.0, 1.0, 1.0), default_ambient, 0.0,1.0);
    Material matte_white = Material(vec3(1.0), default_ambient, 0.0, default_ambient);
    Material orange = Material(vec3(1.0, 0.4, 0.0), default_ambient, 0.0, default_ambient);
    Material orange_glow = orange;
    orange_glow.emissivity = 0.25;

    Material green = Material(vec3(0.0, 1.0, 0.0), default_ambient, 0.65, default_ambient);
    float lulz = 2.0;

    float x_pos = -5.0;
    float y_pos = 0.5;
    x_pos = lulz*pow(sin(iTime),3.0);
    y_pos = lulz/ 16.0 *(16.0 * cos(iTime) - 5.0*cos(2.0*iTime) - 2.0 * cos(3.0*iTime) - cos(4.0*iTime)) + 8.0;

    scene[0] = Object(1, vec3(x_pos, y_pos, 1.0), 1.0, vec3(0.0, 0.0, 0.0), orange_glow);
    scene[1] = Object(1, vec3(0.0, 1.0+sin(iTime*freq), 0.0), 1.0, vec3(0.0, 0.0, 0.0), matte_white);
    scene[2] = Object(2, vec3(0.0, -2.0, 0.0), infinity, vec3(0.0, 1.0, 0.0), matte_white);
    scene[3] = Object(1, vec3(-3.0, 0.5, 0.0), 1.0, vec3(0.0, 0.0, 0.0), green);
    scene[4] = Object(1, vec3(5.0, 1.0, 0.0), 0.5+0.2*sin(iTime*freq), vec3(0.0, 0.0, 0.0), gold);
    scene[5] = Object(1, vec3(5.0*sin(iTime*freq), 2.0+cos(iTime*freq), -5.0), 1.0, vec3(0.0, 0.0, 0.0), white);


    lights[0] = scene[0];
    lights[1] = scene[5];
  }



  void mainImage( out vec4 fragColor, in vec2 fragCoord )
  {
    uv = fragCoord/iResolution.xy;

    view_dir = normalize(vw_dir);
    view_pos = vw_pos;
    make_scene();
    seed=iTime;
    uv = fragCoord;

    float _w = iResolution.x;
    float _h = iResolution.y;

    vec3 ray_dir_from_view = interpolated_ray_dir(uv, _w,_h);

    Ray pixel_ray = Ray(ray_dir_from_view, view_pos, vec3(1.0),1.0);

    vec3 out_col = vec3(0.0);

    for(int k = 0; k< samples; k++){
        pixel_ray = Ray(ray_dir_from_view, view_pos, vec3(1.0),1.0);
        for(int i=0;i<MAX_DEPTH;i++){
            out_col += pixel_ray.energy * trace(pixel_ray, scene, i);
            seed++;
        }
    }
    out_col /= float(samples);
    vec2 UV = fragCoord.xy / iResolution.xy;
    vec4 prev_col = texture2D(iChannel0, UV);
    float weight_prev = 1.0;
    float weight_new = 1.0;
    vec4 new_frag = vec4(weight_prev * prev_col.rgb + weight_new * out_col, prev_col.a+weight_new);
    out_col *= EV;
    vec3 exposure_corrected = out_col * (vec3(1.0) + out_col / vec3(EV*EV)) / (out_col + vec3(1.0));
    vec3 new_col = pow(exposure_corrected, vec3(1.0/GAMMA));


    fragColor = vec4(new_col, 1.0);
  }

  void main() {
  mainImage(gl_FragColor, gl_FragCoord.xy);

  }
  `;

  const main = () => {
    let pointerLocked = false;
    const renderer = new THREE.WebGLRenderer({ canvas });
    renderer.autoClearColor = false;
    let theta = (- 1.0 * Math.PI) / 2.0;
    let phi = Math.PI / 2.0;
    let viewdir = new THREE.Vector3(- 0.0, - 0.0, -1.0);
    const viewpos = new THREE.Vector3(0.0, 4.0, 25.0);
    const M = new THREE.Matrix3().set(1, 0, 0, 0, 1, 0, 0, 0, 1);
    let M2 = new THREE.Matrix4();
    let target = new THREE.Vector3();
    target.addVectors(viewpos, viewdir);
    M2.lookAt(viewpos, target, new THREE.Vector3(0.0, 1.0, 0.0));
    M.setFromMatrix4(M2);
    const sens = 1.0;

    document.addEventListener('keydown', function(event) {
      if (event.keyCode == 87) {
        viewpos.addVectors(
          viewpos,
          new THREE.Vector3(1.0 * viewdir.x, 1.0 * viewdir.y, 1.0 * viewdir.z)
        );
      } else if (event.keyCode == 83) {
        viewpos.addVectors(
          viewpos,
          new THREE.Vector3(
            -1.0 * viewdir.x,
            -1.0 * viewdir.y,
            -1.0 * viewdir.z
          )
        );
      } else if (event.keyCode == 65) {
        viewpos.addVectors(
          viewpos,
          new THREE.Vector3(viewdir.z, 0, -1.0 * viewdir.x)
        );
      } else if (event.keyCode == 68) {
        viewpos.addVectors(
          viewpos,
          new THREE.Vector3(-1.0 * viewdir.z, 0, viewdir.x)
        );
      } else if (event.keyCode == 81) {
        viewpos.addVectors(viewpos, new THREE.Vector3(0.0, 1.0 * sens, 0.0));
      } else if (event.keyCode == 69) {
        viewpos.addVectors(viewpos, new THREE.Vector3(0.0, -1.0 * sens, 0.0));
      }
    });

    canvas.onclick = function() {
      this.requestPointerLock();
    };

    document.addEventListener('pointerlockchange', () => {
      if (document.pointerLockElement === canvas) {
        pointerLocked = true;
      } else {
        pointerLocked = false;
      }
    });

    document.addEventListener('mousemove', e => {
      if (!pointerLocked) return;
      const x = -e.movementY / 500;
      const y = -e.movementX / 500;
      theta -= y;
      phi -= x;
      viewdir = new THREE.Vector3(
        Math.cos(theta) * Math.sin(phi),
        Math.cos(phi),
        Math.sin(theta) * Math.sin(phi)
      );

      M2 = new THREE.Matrix4();
      target = new THREE.Vector3();
      target.addVectors(viewpos, viewdir);
      M2.lookAt(viewpos, target, new THREE.Vector3(0.0, 1.0, 0.0));
      M.setFromMatrix4(M2);
    });

    const camera = new THREE.OrthographicCamera(
      - 1, 1, 1, - 1, - 1, 1
    );
    const scene = new THREE.Scene();
    const plane = new THREE.PlaneBufferGeometry(2, 2);
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));


    const textureSize = 16;
    const dataSize = canvas.clientWidth * canvas.clientHeight;
    const data = new Uint8Array(dataSize);

    for (let i = 0;i < dataSize;i ++) {
          data[i] = Math.round(Math.random() * 255);
    }
    const updateTexture = () => {
      for (let i = 0;i < dataSize;i ++) {
        data[i] = Math.round(Math.random() * 255);
      }
    };

    const texture = new THREE.DataTexture(
      data,
      textureSize,
      textureSize,
      THREE.LUMINACE,
      THREE.UnsignedByteType
    );

    const uniforms = {
      samples: {value: samples},
      iTime: { value: 1 },
      iResolution: { value: new THREE.Vector3() },
      iChannel0: { value: texture },
      vw_dir: { value: viewdir },
      vw_pos: { value: viewpos },
      camera_matrix: { value: M }
    };

    const material = new THREE.ShaderMaterial({
      fragmentShader,
      uniforms
    });

    scene.add(new THREE.Mesh(plane, material));

    const resizeRendererToDisplaySize = (composer, renderer) => {
      const canvas = renderer.domElement;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const needResize = canvas.width !== width || canvas.height !== height;
      if (needResize) {
        composer.setSize(width, height, false);
        renderer.setSize(width, height, false);
        updateTexture();
      }
      return needResize;
    };

    const d0 = new Date();
    let t0 = d0.getTime();
    const render = time => {
      const d1 = new Date();
      time = (d1.getTime() - d0.getTime()) * 0.001;
      document.querySelector('.fps').textContent = `FPS: ${Math.round(1000.0 / (d1.getTime() - t0))}`;
      t0 = d1.getTime();

      resizeRendererToDisplaySize(composer, renderer);

      const canvas = renderer.domElement;
      uniforms.iResolution.value.set(canvas.width, canvas.height, 1);
      uniforms.iTime.value = time;
      uniforms.iChannel0.value = texture;
      uniforms.vw_dir.value = viewdir;
      uniforms.vw_pos.value = viewpos;
      uniforms.camera_matrix.value = M;
      uniforms.samples.value = samples;

      composer.render(scene, camera);

      requestAnimationFrame(render);
    };

    requestAnimationFrame(render);
  };

  main();


}
