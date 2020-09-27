import './style.css';
import * as THREE from './three/three.module.js';

{
  const canvas = document.querySelector('#c');
  const renderer = new THREE.WebGLRenderer({canvas});
  renderer.autoClearColor = false;
  const textureSize = 16;
  const dataSize = 10;
  const data = new Uint8Array(dataSize);

  const camera = new THREE.OrthographicCamera(
    - 1, // left
    1, // right
    1, // top
    - 1, // bottom
    - 1, // near,
    1 // far
  );

  const fragmentShader = `
		#include <common>

		uniform vec3 iResolution;
		uniform float iTime;
		uniform sampler2D iChannel0;

		#define M_PI 3.1415926535897932384626433832795
		const float infinity = 1.0 / 0.000000000001;
		vec3 view_dir = vec3(-0.1, -0.1, -1.0);
		vec3 view_pos = vec3(0.0, 4.0, 25.0);

		const int MAX_DEPTH = 3;
		const int NUM_OF_OBJECTS = 6;
		const int NUM_OF_LIGHTS = 2;
		const int samples = 20;
		const float GAMMA = 2.2;


		const float bias = 0.000001;
		const float minimum_collision_distance = 0.00001;
		const float fov = 60.0 * M_PI / 180.0;

		const float EV = 10.0;
		const float light_intensity_scale = 1.0;
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
			//type 0: circle
			//type 1: sphere
			//type 2: plane
			//type 3: triangle
			vec3 normal;

			Material mat;

		};

		float seed = 0.0;
		vec2 uv = vec2(0.0);

		float random() {
			return fract(sin(dot(uv, mat2(cos(iTime), sin(iTime), -sin(iTime), cos(iTime)) * vec2( 11.9898, 78.233)) + seed++) * 43758.5453);
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

			//rotate to normal plane

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
			float x = (2.0 * (_xy.x +0.5)/w - 1.0);
			float y = -(1.0-2.0*(_xy.y+0.5)/h)/aspect_ratio;

			float D = 1.0 / tan(fov/2.0);

			return normalize(vec3(x,y,0) + D * view_dir);
		}

		vec3 get_point_at_distance(Ray _ray, float d){
			return _ray.m_or + d * _ray.m_dir;
		}


		float intersect(Ray _ray, Object _obj){
			float d = infinity;
			if (_obj.type == 0 ) // circle
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



					//random ray towards light
					vec3 diffuse_direct_col = vec3(0.0);
					for(int k =0;k<1;k++){
						for(int i=0;i<NUM_OF_OBJECTS;i++){
							//make towards an object if it is emissive
							if(_scene[i].mat.emissivity > 0.0){
								//Cast a ray towards an object that is emissive
								vec3 pt_to_light_vec0 = normalize(_scene[i].pos - intersection_point);

								vec3 random_dir = diff_cos_weighted_direction(light_directionality, pt_to_light_vec0);

								Ray r_to_light = create_new_ray(random_dir, intersection_point + bias * surf_norm);

								vec2 towards_light_intersection_info = intersect_scene(r_to_light, _scene);
								int index_collision = int(towards_light_intersection_info.x);
								// The ray has collided with something, let's check its distance and the emissivity of the collided object

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

					//reflect
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
			float freq = 5.0;
			//always add glowing object to lights
			//material: color amb spec em
			//white light
			Material white = Material(vec3(1.0, 1.0, 1.0), default_ambient, 0.0,1.0);
			//matte white
			Material matte_white = Material(vec3(1.0), default_ambient, 0.0, default_ambient);
			//glowing orange
			Material orange = Material(vec3(1.0, 0.4, 0.0), default_ambient, 0.0, default_ambient);
			//teal glowing
			float glow = 0.5*cos(freq*iTime)+0.5;
			Material teal = Material(vec3(0.0, 1.0, 1.0), default_ambient, 0.5, default_ambient);
			Material gold = Material(vec3(0.83, 0.68, 0.216), default_ambient, 1.0, default_ambient);
			Material orange_glow = orange;
			orange_glow.emissivity = 0.05;

			Material green = Material(vec3(0.0, 1.0, 0.0), default_ambient, 0.65, default_ambient);
			//object
			//type pos radius normal material
			float lulz = 1.0;

			float x_pos = -5.0;
			float y_pos = 0.5;
			//x_pos = lulz*pow(sin(iTime),3.0);
			//y_pos = lulz/ 16.0 *(16.0 * cos(iTime) - 5.0*cos(2.0*iTime) - 2.0 * cos(3.0*iTime) - cos(4.0*iTime)) + 8.0;

			scene[0] = Object(1, vec3(x_pos, y_pos, 1.0), 1.0, vec3(0.0, 0.0, 0.0), orange_glow);
			scene[1] = Object(1, vec3(0.0, 0.5, 0.0), 1.0, vec3(0.0, 0.0, 0.0), matte_white);
			scene[2] = Object(2, vec3(0.0, -0.5, 0.0), infinity, vec3(0.0, 1.0, 0.0), matte_white);
			scene[3] = Object(1, vec3(-2.0, 0.5, 0.0), 1.0, vec3(0.0, 0.0, 0.0), green);
			scene[4] = Object(1, vec3(2.0, 0.5, 0.0), 1.0, vec3(0.0, 0.0, 0.0), gold);
			scene[5] = Object(1, vec3(4.0, 2.0, -5.0), 1.0, vec3(0.0, 0.0, 0.0), white);


			lights[0] = scene[0];
			lights[1] = scene[5];
		}



		void mainImage( out vec4 fragColor, in vec2 fragCoord )
		{
			// Normalized pixel coordinates (from 0 to 1)
			vec2 uv = fragCoord/iResolution.xy;

			view_dir = normalize(view_dir);
			// process_input();
			make_scene();
			// Normalized pixel coordinates (from 0 to 1)UV = fragCoord;
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
			// blend with previous
			vec2 UV = fragCoord.xy / iResolution.xy;
			vec4 prev_col = texture2D(iChannel0, UV);
			float weight_prev = 1.0;
			float weight_new = 1.0;
			vec4 new_frag = vec4(weight_prev * prev_col.rgb + weight_new * out_col, prev_col.a+weight_new);


			fragColor = vec4(out_col,1.0);
			fragColor = new_frag;
		}

		void main() {
		mainImage(gl_FragColor, gl_FragCoord.xy);

		}
		`;

  const resizeRendererToDisplaySize = renderer => {
    const canvas = renderer.domElement;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const needResize = canvas.width !== width || canvas.height !== height;
    if (needResize) {
      renderer.setSize(width, height, false);
    }
    return needResize;
  };

  const init = () => {
    const scene = new THREE.Scene();
    const plane = new THREE.PlaneBufferGeometry(2, 2);
    const texture = new THREE.DataTexture(
      data,
      textureSize,
      textureSize,
      THREE.LUMINACE,
      THREE.UnsignedByteType
    );

    const uniforms = {
      iTime: { value: 0 },
      iResolution: { value: new THREE.Vector3() },
      iChannel0: { value: texture }
    };

    const material = new THREE.ShaderMaterial({
      fragmentShader,
      uniforms
    });

    scene.add(new THREE.Mesh(plane, material));
    const d0 = new Date();


    const render = (time) => {
      const d1 = new Date();
      time = (d1.getTime() - d0.getTime()) * 0.001; // convert to seconds
      console.log(time);
      resizeRendererToDisplaySize(renderer);

      const canvas = renderer.domElement;
      uniforms.iResolution.value.set(canvas.width, canvas.height, 1);
      uniforms.iTime.value = time;

      renderer.render(scene, camera);

      requestAnimationFrame(render);
    };
    requestAnimationFrame(render);
  };

  init();
}
