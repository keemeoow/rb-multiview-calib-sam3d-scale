# SAM 3D Objects — GB10 (ARM64 / Blackwell sm_121 / CUDA 13) 셋업

`third_party/sam-3d-objects` **추론** 스택을 NVIDIA **GB10**(ARM64 aarch64 + Blackwell, CUDA 13) 환경으로 포팅한 기록.

> ⚠️ 공식 setup 전제는 **Linux x86-64 + 32GB+ NVIDIA GPU + CUDA 12.1**. 우리 머신은 **ARM64 + CUDA 13**이라 공식 절차(`environments/default.yml`, micromamba)를 **그대로 쓸 수 없어** 아래처럼 다시 구성했다.

---

## 1. repo에 SAM3D 추가

```bash
mkdir -p third_party
git clone https://github.com/facebookresearch/sam-3d-objects.git third_party/sam-3d-objects

# 설치 기준점(commit) 기록 — 버전 일치용
mkdir -p repro
git -C third_party/sam-3d-objects rev-parse HEAD | tee repro/sam3d_git_commit.txt
```

## 2. (선택) 출력 경로 `/workspace` 연결

`sudo`로 만들면 root 소유가 되어 출력 쓰기 권한 문제가 생기니 마지막에 `chown`으로 본인 소유로 돌린다. 입력 폴더는 **미리 생성**해야 심볼릭 링크가 깨지지 않는다.

```bash
mkdir -p data/sam3d_inputs data/sam3d_outputs/native_glb
sudo mkdir -p /workspace
sudo ln -sfn "$PWD/third_party/sam-3d-objects"     /workspace/sam-3d-objects
sudo ln -sfn "$PWD/data/sam3d_inputs"              /workspace/sam3d_inputs
sudo ln -sfn "$PWD/data/sam3d_outputs/native_glb"  /workspace/output_native_glb
sudo chown -R "$USER:$USER" data /workspace
```

## 3. conda 빈 환경 생성 (의존성은 4번에서 직접 설치)

```bash
conda create -n sam3d python=3.11 -y
conda activate sam3d
```

## 4. 최종 환경 세팅 (검증 완료)

| 구성요소 | 상태 | 비고 |
|---|---|---|
| torch 2.9.1+cu130 | ✅ GPU 연산 검증 | sm_120 PTX → sm_121 JIT |
| torchvision 0.24.1 | ✅ CUDA op 검증 | |
| pytorch3d 0.7.9 | ✅ 소스 빌드, knn GPU 검증 | |
| kaolin 0.18 | ✅ 소스 빌드 (usd-core 제외) | |
| cumm + spconv 2.3.8 | ✅ sparse conv GPU 검증 | **CUDA13 c++17 + GB10 arch 패치 (최대 난관)** |
| gsplat 1.5.3 | ✅ rasterization GPU 검증 | |
| moge / utils3d(핀 커밋) / 기타 | ✅ 설치 | timm·lightning·optree·xatlas 등 |
| `from inference import Inference` | ✅ 전체 import OK | sdpa attention, spconv 백엔드 |

- xformers·flash_attn은 `ATTN_BACKEND=sdpa`로 대체(빌드 회피), diff_gaussian은 선택사항이라 생략.
- 매 세션 활성화는 repo 루트의 **`source sam3d_env_gb10.sh`** 한 줄 (conda activate + PATH + CUDA_HOME + ATTN_BACKEND + arch list 설정).

### 실제 설치 명령 (재현용 기록 — 이 머신엔 이미 적용됨)

```bash
conda activate sam3d
export PATH="$HOME/miniconda3/envs/sam3d/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="12.0+PTX" MAX_JOBS=10

# 빌드 도구 — setuptools는 반드시 <81 (82는 pkg_resources 제거 → kaolin 빌드 실패)
pip install -U pip ninja "setuptools<81" wheel Cython pybind11 pccm ccimport fire fvcore iopath

# torch / torchvision (cu130, aarch64) + numpy
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
pip install "numpy<2.3"

# 순수 파이썬 추론 deps
pip install omegaconf "hydra-core==1.3.2" einops einops-exts matplotlib seaborn gradio \
  trimesh scipy scikit-image opencv-python-headless imageio tqdm rich loguru easydict roma rootutils \
  timm astor igraph lightning optree pymeshfix pyvista xatlas open3d \
  wget ipycanvas ipyevents plyfile pygltflib warp-lang

# utils3d / moge — 반드시 핀 커밋 (PyPI·최신판엔 utils3d.numpy.depth_edge 없음)
pip install --no-deps "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"
pip install --no-deps "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

# pytorch3d (소스 빌드)
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

# kaolin (소스, --no-deps: usd-core는 aarch64 휠 없음 / kaolin이 graceful 처리)
pip install --no-build-isolation --no-deps "git+https://github.com/NVIDIAGameWorks/kaolin.git"
pip install wget ipycanvas ipyevents plyfile pybind11 pygltflib warp-lang   # kaolin 런타임 deps

# cumm + spconv (소스, editable, JIT) — /tmp 말고 영구 경로에 보관
mkdir -p ~/sam3d_src
git clone https://github.com/FindDefinition/cumm.git ~/sam3d_src/cumm
git clone https://github.com/traveller59/spconv.git  ~/sam3d_src/spconv
export CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST="12.0+PTX"
#  ★ CUDA 13 핵심 패치 1: JIT 기본 C++ 표준 14→17 (libcu++/cccl가 c++17 요구)
sed -i 's/std: str = "c++14"/std: str = "c++17"/g' ~/sam3d_src/cumm/cumm/nvrtc/__init__.py
#  ★ spconv/spconv/build.py 의 build_pybind(cus, ...) 호출에 std="c++17", 한 줄 추가 (수동 편집)
#  ★ CUDA 13 핵심 패치 2 (GB10 arch): GB10이 cumm GPU 인식표에 없어 CUMM_CUDA_ARCH_LIST가
#    빌드 경로에 안 닿으면 옛 arch(compute_52..86)로 폴백 → CUDA13 nvcc가
#    'Unsupported gpu architecture compute_52'로 거부. GB10을 인식표에 추가 + 폴백 arch도 12.0으로.
sed -i 's#_GPU_NAME_TO_ARCH = {#_GPU_NAME_TO_ARCH = {\n    r"(NVIDIA )?GB10": "120",#' ~/sam3d_src/cumm/cumm/common.py
sed -i 's#_all_arch = "5.2;6.0;6.1;7.0;7.5;8.0;8.6+PTX"#_all_arch = "12.0+PTX"#' ~/sam3d_src/cumm/cumm/common.py
pip install --no-build-isolation -e ~/sam3d_src/cumm
pip install --no-build-isolation --no-deps -e ~/sam3d_src/spconv

# gsplat
pip install gsplat
```

> **핵심 함정 4가지**: ① `setuptools<81`  ② `utils3d` 핀 커밋  ③ cumm/spconv **c++17 패치**  ④ cumm **GB10 arch 패치**(compute_52 폴백 방지).
> cumm/spconv/gsplat은 첫 실행 때 GPU 커널을 **JIT 컴파일**하므로, 런타임에도 `ninja`+`nvcc`(PATH)와 `CUDA_HOME`이 필요하다(=`sam3d_env_gb10.sh`가 처리). cumm/spconv 소스(`~/sam3d_src`)는 editable 설치라 **지우면 안 됨**.

## 5. 체크포인트 다운로드 → demo 실행

`third_party/sam-3d-objects` 에서 진행. ①②는 한 번만, 이후엔 환경 활성화 후 `python demo.py`만.

**① inference.py CUDA_HOME 패치** (런타임 JIT가 nvcc를 찾도록; 원본은 CUDA_HOME을 conda 경로로 덮어씀)
```bash
sed -i 's#os.environ\["CUDA_HOME"\] = os.environ\["CONDA_PREFIX"\]#os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME", "/usr/local/cuda")#' notebook/inference.py
grep -n CUDA_HOME notebook/inference.py | head -1
```

**② 체크포인트 다운로드** (gated repo — HF 로그인 + 접근 승인 필요. 실제 **~12GB / 28파일**)

> ⚠️ **이 박스에서 `hf download`는 멈춘다(hang, 0B/s).** 원인 두 가지가 겹침: ㉠ IPv6 경로가 죽어 있는데 DNS가 IPv6를 먼저 줘서 hf가 죽은 소켓(`CLOSE-WAIT`)을 붙잡고 무한 대기, ㉡ huggingface_hub 1.13.0(httpx)이 끊긴 연결을 복구 못 하는 버그. → **시스템 IPv4 우선 + `curl -4` 직접 다운로드**로 해결. (`hf download`의 "5.98G"는 목록을 다 못 센 과소표시였고 실제는 ~12GB)

```bash
# ㉠ 시스템 IPv4 우선 (1회, sudo) — 모든 앱에 적용. 즉시 반영(재부팅 불필요)
echo 'precedence ::ffff:0:0/96 100' | sudo tee -a /etc/gai.conf

# ㉡ curl -4 직접 다운로드 — repo에 포함된 스크립트 사용
#    third_party/sam-3d-objects/curl_download.sh : HF API로 파일목록 받아 curl -4 -C-(이어받기)
#    + --speed-limit/--speed-time(느려지면 끊고 재시도)로 끊김 자가복구. 검증 ~7MB/s.
bash curl_download.sh          # → checkpoints/hf-download/checkpoints/*.ckpt
mv checkpoints/hf-download/checkpoints checkpoints/hf
rm -rf checkpoints/hf-download
ls checkpoints/hf              # pipeline.yaml 보이면 성공
```

**③ 실행** (첫 실행은 spconv GEMM 커널 JIT 컴파일로 수 분 소요, 이후 캐시됨)
```bash
python demo.py 2>&1 | tee demo_run.log   # 성공 시 splat.ply 생성
```

> ⚠️ demo.py는 실행 중 **MoGe**(`Ruicheng/moge-vitl` model.pt, ~1.26GB)를 HF 캐시로 받는데, 이것도 위 hf hang에 걸린다. 멈추면(`model.pt 16%...`에서 정지) demo.py 종료 후 `curl -4`로 캐시 blob을 직접 완성:
> ```bash
> BLOB=~/.cache/huggingface/hub/models--Ruicheng--moge-vitl/blobs/da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f
> curl -4 -L -C - --speed-limit 51200 --speed-time 15 --retry 100 --retry-all-errors \
>   "https://huggingface.co/Ruicheng/moge-vitl/resolve/main/model.pt" -o "$BLOB.incomplete"
> ls -l "$BLOB.incomplete"   # 1256823446 바이트 확인 후
> mv "$BLOB.incomplete" "$BLOB"
> ```
> 이후 demo.py 재실행 시 캐시 히트로 다운로드를 건너뛴다.

**✅ 검증 완료 (2026-05-26):** GB10에서 `python demo.py` 끝까지 성공 → `splat.ply` 생성 (55MB, binary PLY, 가우시안 **842,112점**).

### 막힐 때
- **`nvcc fatal: Unsupported gpu architecture 'compute_52'`** (spconv JIT 빌드 중): GB10이 cumm GPU 인식표에 없어 옛 arch로 폴백. → 4번의 cumm `common.py` **GB10 arch 패치** 적용 후 `rm -rf ~/sam3d_src/spconv/spconv/build` 하고 재실행. (검증: `_get_cuda_arch_flags()`가 `compute_120`만 내면 OK)
- **다운로드(`hf download`/MoGe)가 멈춤 (0B/s, `CLOSE-WAIT` 소켓)**: IPv6 죽음 + hf 1.13.0 버그. → `/etc/gai.conf` IPv4 우선 + `curl -4` 직접 받기(②③ 참고). 진단: `curl -4` vs `curl -6`로 같은 파일 받아보면 IPv6가 즉시 실패(exit 7)하는 걸로 바로 갈림.
- `ModuleNotFoundError: nvdiffrast` (메시 추출 경로에서만): `pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"`
- `libcu++ requires at least C++ 17`: 해당 확장의 JIT가 c++14 사용 → c++17로 패치 필요
- `nvcc not found` / `cannot find -lcuda`: `source sam3d_env_gb10.sh` 또는 ① 패치 누락
