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
| cumm + spconv 2.3.8 | ✅ sparse conv GPU 검증 | **CUDA13 c++17 패치 (최대 난관)** |
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
#  ★ CUDA 13 핵심 패치: JIT 기본 C++ 표준 14→17 (libcu++/cccl가 c++17 요구)
sed -i 's/std: str = "c++14"/std: str = "c++17"/g' ~/sam3d_src/cumm/cumm/nvrtc/__init__.py
#  ★ spconv/spconv/build.py 의 build_pybind(cus, ...) 호출에 std="c++17", 한 줄 추가 (수동 편집)
pip install --no-build-isolation -e ~/sam3d_src/cumm
pip install --no-build-isolation --no-deps -e ~/sam3d_src/spconv

# gsplat
pip install gsplat
```

> **핵심 함정 3가지**: ① `setuptools<81`  ② `utils3d` 핀 커밋  ③ cumm/spconv **c++17 패치**.
> cumm/spconv/gsplat은 첫 실행 때 GPU 커널을 **JIT 컴파일**하므로, 런타임에도 `ninja`+`nvcc`(PATH)와 `CUDA_HOME`이 필요하다(=`sam3d_env_gb10.sh`가 처리). cumm/spconv 소스(`~/sam3d_src`)는 editable 설치라 **지우면 안 됨**.

## 5. 체크포인트 다운로드 → demo 실행

`third_party/sam-3d-objects` 에서 진행. ①②는 한 번만, 이후엔 환경 활성화 후 `python demo.py`만.

**① inference.py CUDA_HOME 패치** (런타임 JIT가 nvcc를 찾도록; 원본은 CUDA_HOME을 conda 경로로 덮어씀)
```bash
sed -i 's#os.environ\["CUDA_HOME"\] = os.environ\["CONDA_PREFIX"\]#os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME", "/usr/local/cuda")#' notebook/inference.py
grep -n CUDA_HOME notebook/inference.py | head -1
```

**② 체크포인트 다운로드** (gated repo — HF 로그인 + 접근 승인 필요)
```bash
hf download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects
mv checkpoints/hf-download/checkpoints checkpoints/hf
rm -rf checkpoints/hf-download
ls checkpoints/hf      # pipeline.yaml 보이면 성공
```

**③ 실행** (첫 실행은 JIT 컴파일로 수 분 소요, 이후 캐시됨)
```bash
python demo.py         # 성공 시 splat.ply 생성
```

### 막힐 때
- `ModuleNotFoundError: nvdiffrast` (메시 추출 경로에서만): `pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"`
- `libcu++ requires at least C++ 17`: 해당 확장의 JIT가 c++14 사용 → c++17로 패치 필요
- `nvcc not found` / `cannot find -lcuda`: `source sam3d_env_gb10.sh` 또는 ① 패치 누락
