# Hybrid Structure-from-Motion: Daily Implementation Tracker

## 🎯 Project Overview

**Goal**: Build a hybrid Structure-from-Motion system that combines points, lines, and vanishing points for robust 3D reconstruction, especially in challenging indoor environments.

**Based on Paper**: "Robust Incremental Structure-from-Motion with Hybrid Features" (Liu et al.)

**Target Performance**: 20-40% improvement over COLMAP on poorly textured scenes.

---

## 🛠️ Environment Setup (COMPLETED)

- **OS**: Ubuntu 22.04 LTS (switched from Windows)
- **GPU**: RTX 4070 Laptop (8GB VRAM)
- **CUDA**: 12.4
- **OpenCV**: 4.13.0-dev with CUDA support
- **Python**: 3.10.12 with virtual environment
- **Compiler**: GCC/G++ 11.4.0

**Environment Status**: ✅ READY

---

## 📅 Daily Implementation Schedule (35 Days)

### **WEEK 1: Foundation & Basic Infrastructure**

#### **Day 1: Project Structure & Build System**

**Objective**: Set up complete project structure and build system
**Deliverables**:

- Complete CMakeLists.txt with all dependencies
- vcpkg.json with package specifications
- Python package structure with pybind11 bindings
- Basic C++ data types (Camera, Image, Point2D, Point3D)

**LinkedIn Post**: "🚀 Starting my hybrid Structure-from-Motion project! Setting up a robust C++/Python architecture for combining points, lines, and vanishing points in 3D reconstruction. #ComputerVision #SfM #OpenCV"

**Key Files to Create**:

```
hybrid_sfm/
├── CMakeLists.txt
├── vcpkg.json
├── setup.py
├── src/core/types/
│   ├── camera.h/cpp
│   ├── image.h/cpp
│   └── features.h/cpp
└── python/hybrid_sfm/__init__.py
```

**Success Metric**: Successfully compile and import Python module

---

#### **Day 2: Camera Models & Calibration**

**Objective**: Implement camera intrinsic/extrinsic models
**Deliverables**:

- Pinhole camera model with distortion
- Camera pose representation (rotation + translation)
- COLMAP camera file parser
- Unit tests for camera operations

**LinkedIn Post**: "📷 Implemented robust camera models for my hybrid SfM system. Supporting pinhole cameras with distortion coefficients - the foundation for accurate 3D reconstruction! #CameraCalibration #3DReconstruction"

**Key Implementation**:

- Quaternion-based rotations
- Camera matrix operations
- Projection/unprojection functions
- COLMAP format compatibility

**Success Metric**: Parse COLMAP camera files and perform accurate projections

---

#### **Day 3: Image Data Pipeline**

**Objective**: Create image loading and preprocessing pipeline
**Deliverables**:

- Multi-format image loader (supports common formats)
- Image pyramid generation for multi-scale processing
- Metadata extraction and management
- Memory-efficient image containers

**LinkedIn Post**: "🖼️ Built a robust image processing pipeline for hybrid SfM! Multi-scale image pyramids and efficient memory management - ready for feature extraction. #ImageProcessing #OpenCV"

**Key Features**:

- CUDA-accelerated image operations
- Lazy loading for large datasets
- Image undistortion pipeline
- Thread-safe image management

**Success Metric**: Load and process 100+ images efficiently

---

#### **Day 4: Point Feature Detection (SIFT)**

**Objective**: Implement CUDA-accelerated SIFT detection
**Deliverables**:

- SIFT feature detector wrapper
- GPU memory management for features
- Feature quality filtering
- Visualization tools for detected features

**LinkedIn Post**: "⚡ CUDA-accelerated SIFT detection is live! Extracting thousands of robust keypoints at lightning speed for my hybrid SfM system. GPU power makes all the difference! #CUDA #FeatureDetection"

**Performance Target**: >10x speedup over CPU SIFT

**Key Implementation**:

- OpenCV CUDA SIFT integration
- Adaptive feature count per image
- Octave-based processing
- Feature response thresholding

**Success Metric**: Detect 2000+ features per image in <100ms

---

#### **Day 5: Point Feature Matching**

**Objective**: Implement robust feature matching with CUDA
**Deliverables**:

- FLANN-based feature matching
- Lowe's ratio test implementation
- Cross-checking for robust matches
- Match visualization tools

**LinkedIn Post**: "🔗 Robust feature matching implemented! Lowe's ratio test + cross-checking ensures high-quality matches. Watch those feature correspondences light up! #FeatureMatching #ComputerVision"

**Key Features**:

- GPU-accelerated matching
- Geometric verification
- Match filtering and ranking
- Batch processing for multiple image pairs

**Success Metric**: 500+ reliable matches between image pairs

---

#### **Day 6: Geometric Verification (Essential Matrix)**

**Objective**: Implement RANSAC-based geometric verification
**Deliverables**:

- 5-point algorithm for essential matrix
- RANSAC implementation with adaptive thresholds
- Pose recovery from essential matrix
- Inlier/outlier classification

**LinkedIn Post**: "📐 Geometric verification complete! RANSAC + 5-point algorithm filters out bad matches and recovers camera poses. The math behind SfM is beautiful! #RANSAC #EssentialMatrix"

**Key Implementation**:

- OpenCV's findEssentialMat with CUDA
- Chirality check for pose disambiguation
- Confidence-based termination
- Uncertainty estimation

**Success Metric**: >90% inlier rate on good image pairs

---

#### **Day 7: Basic SfM Pipeline Integration**

**Objective**: Integrate components into working point-based SfM
**Deliverables**:

- End-to-end point-based reconstruction
- Two-view reconstruction demo
- Basic bundle adjustment integration
- Performance benchmarking

**LinkedIn Post**: "🎉 First milestone reached! Basic point-based SfM pipeline is working. Two-view reconstruction with bundle adjustment - the foundation is solid! #SfM #Milestone"

**Demo Output**: 3D point cloud from two images
**Performance Target**: Process image pair in <5 seconds

---

### **WEEK 2: Line Feature Integration**

#### **Day 8: Line Detection (DeepLSD)**

**Objective**: Integrate DeepLSD for robust line detection
**Deliverables**:

- DeepLSD model integration via ONNX
- Line segment extraction and filtering
- Multi-scale line detection
- Line quality assessment

**LinkedIn Post**: "🔍 Deep learning meets traditional computer vision! Integrated DeepLSD for robust line detection. Even in challenging lighting, we're finding those structural lines! #DeepLearning #LineDetection"

**Key Features**:

- ONNX Runtime integration
- GPU inference acceleration
- Line segment merging
- Angle and length filtering

**Success Metric**: Detect 50+ reliable lines per indoor image

---

#### **Day 9: Line Matching (GlueStick)**

**Objective**: Implement GlueStick for line matching
**Deliverables**:

- GlueStick model integration
- Line descriptor computation
- Robust line matching pipeline
- Match quality scoring

**LinkedIn Post**: "🧲 GlueStick line matching is incredible! Matching line segments across views with neural network precision. The future of feature matching is here! #NeuralMatching #LineMatching"

**Key Implementation**:

- Joint point-line matching
- Geometric consistency checks
- Match confidence scoring
- Efficient batch processing

**Success Metric**: 20+ reliable line matches between image pairs

---

#### **Day 10: Vanishing Point Detection**

**Objective**: Implement vanishing point detection using JLinkage
**Deliverables**:

- JLinkage clustering algorithm
- Line grouping by vanishing points
- Manhattan world assumptions
- VP quality assessment

**LinkedIn Post**: "📍 Vanishing points detected! JLinkage algorithm groups parallel lines and finds those crucial vanishing points. Manhattan world structure emerging! #VanishingPoints #StructuralCV"

**Key Features**:

- Multi-VP detection (typically 3 for Manhattan)
- Outlier line rejection
- Confidence-based VP selection
- Visualization of VP lines

**Success Metric**: Detect 2-3 reliable VPs per structured scene

---

#### **Day 11: Hybrid Feature Matching**

**Objective**: Create unified point-line-VP matching framework
**Deliverables**:

- Cross-modal feature verification
- Geometric consistency between feature types
- Unified match data structures
- Joint optimization preparation

**LinkedIn Post**: "🔄 Hybrid feature matching achieved! Points, lines, and vanishing points working together for robust correspondences. The synergy is remarkable! #HybridFeatures #MultiModal"

**Key Implementation**:

- Point-line spatial relationships
- Line-VP consistency checks
- Multi-feature RANSAC
- Uncertainty weighting

**Success Metric**: Reliable hybrid matches on challenging indoor scenes

---

#### **Day 12: Line Triangulation**

**Objective**: Implement incremental line triangulation
**Deliverables**:

- 3D line representation (Plücker coordinates)
- Two-view line triangulation
- Multi-view line optimization
- Degenerate configuration handling

**LinkedIn Post**: "📏 3D line triangulation working! Converting 2D line segments into 3D space using Plücker coordinates. The geometry is getting richer! #LineTriangulation #PluckerCoordinates"

**Key Features**:

- Robust line fitting in 3D
- Endpoint optimization
- Uncertainty propagation
- Geometric validation

**Success Metric**: Triangulate 10+ 3D lines from multi-view data

---

#### **Day 13: Vanishing Point Triangulation**

**Objective**: Implement VP triangulation and direction recovery
**Deliverables**:

- Single-view VP to 3D direction
- Multi-view VP consistency
- Manhattan frame recovery
- Orthogonality constraints

**LinkedIn Post**: "🧭 3D vanishing point triangulation complete! Recovering the 3D directions of parallel line families. Manhattan world constraints are powerful! #VanishingPoints #ManhattanWorld"

**Key Implementation**:

- Spherical representation of directions
- Cross-view VP association
- Orthogonal constraint enforcement
- Reference frame alignment

**Success Metric**: Recover consistent 3D Manhattan frame

---

#### **Day 14: Hybrid Geometric Verification**

**Objective**: Extend RANSAC for multi-feature geometric verification
**Deliverables**:

- Multi-solver RANSAC framework
- Point-line PnP solvers
- VP-constrained pose estimation
- Hybrid inlier counting

**LinkedIn Post**: "🎯 Hybrid RANSAC is game-changing! Using points, lines, AND vanishing points for camera pose estimation. Robust estimation just got way more robust! #RANSAC #HybridGeometry"

**Key Features**:

- P3P, P2P1L, P1P2L solvers
- VP-gravity alignment
- Adaptive solver selection
- Multi-feature scoring

**Success Metric**: Successful pose estimation with <20 point features

---

### **WEEK 3: Bundle Adjustment & Optimization**

#### **Day 15: Cost Function Design**

**Objective**: Implement cost functions for hybrid bundle adjustment
**Deliverables**:

- Point reprojection error
- Line-to-line distance metrics
- VP direction consistency
- Uncertainty-weighted costs

**LinkedIn Post**: "⚡ Designing cost functions for hybrid bundle adjustment! Balancing point accuracy, line distances, and VP consistency. Mathematical beauty meets practical optimization! #BundleAdjustment #Optimization"

**Key Implementation**:

- Ceres Solver integration
- Automatic differentiation
- Robust loss functions
- Multi-scale optimization

**Success Metric**: Well-conditioned optimization problems

---

#### **Day 16: Uncertainty Propagation**

**Objective**: Implement analytical uncertainty propagation for lines
**Deliverables**:

- Jacobian computation for line residuals
- Covariance propagation
- Uncertainty visualization
- Quality metrics based on uncertainty

**LinkedIn Post**: "📊 Breakthrough in uncertainty estimation! First analytical method for propagating uncertainties in 3D line optimization. Now we know which features to trust! #UncertaintyQuantification #Innovation"

**Key Features**:

- Second-order sensitivity analysis
- Line-specific uncertainty models
- Feature reliability scoring
- Uncertainty-based filtering

**Success Metric**: Uncertainty correlates with ground-truth errors

---

#### **Day 17: Two-Step Bundle Adjustment**

**Objective**: Implement reliable/unreliable track separation
**Deliverables**:

- Track reliability classification
- Two-phase optimization
- Inactive support caching
- Adaptive track management

**LinkedIn Post**: "🎯 Two-step bundle adjustment implemented! Separating reliable from unreliable tracks prevents bad features from corrupting the entire reconstruction. Smart optimization! #BundleAdjustment #TrackManagement"

**Key Implementation**:

- Uncertainty-based track classification
- Fixed-pose refinement for unreliable tracks
- Dynamic track promotion/demotion
- Memory-efficient caching

**Success Metric**: Stable optimization on challenging sequences

---

#### **Day 18: Local Bundle Adjustment**

**Objective**: Implement sliding-window local BA
**Deliverables**:

- Keyframe selection strategy
- Local BA window management
- Marginalization of old keyframes
- Loop closure detection preparation

**LinkedIn Post**: "🪟 Sliding-window bundle adjustment is working! Local optimization keeps the system efficient while maintaining global consistency. Scalable SfM architecture! #LocalBA #Scalability"

**Key Features**:

- Uncertainty-based keyframe selection
- Efficient sparse matrix operations
- Incremental Cholesky updates
- Multi-threading support

**Success Metric**: Real-time performance on 10-frame windows

---

#### **Day 19: Global Bundle Adjustment**

**Objective**: Implement full global optimization
**Deliverables**:

- Full problem formulation
- Sparse matrix optimization
- Convergence monitoring
- Performance profiling

**LinkedIn Post**: "🌍 Global bundle adjustment complete! Optimizing the entire reconstruction simultaneously - thousands of parameters converging beautifully. The power of sparse optimization! #GlobalBA #Optimization"

**Key Implementation**:

- Ceres Solver configuration
- Parallel Jacobian computation
- Convergence criteria
- Memory optimization

**Success Metric**: Convergence on 100+ image datasets

---

#### **Day 20: Structural Constraints**

**Objective**: Implement point-line and line-VP associations
**Deliverables**:

- Point-on-line constraints
- Line-VP parallelism constraints
- Manhattan world enforcement
- Constraint weighting

**LinkedIn Post**: "🏗️ Structural constraints are revolutionary! Points lying on lines, parallel line families, orthogonal VPs - injecting geometric knowledge into optimization! #StructuralConstraints #GeometricCV"

**Key Features**:

- Constraint residual functions
- Adaptive constraint weighting
- Constraint violation detection
- Geometric consistency enforcement

**Success Metric**: Cleaner, more structured reconstructions

---

#### **Day 21: Optimization Integration Test**

**Objective**: Integrate all optimization components
**Deliverables**:

- End-to-end optimization pipeline
- Performance benchmarking
- Quality metrics
- Comparison with point-only BA

**LinkedIn Post**: "🎯 Hybrid optimization pipeline complete! Points, lines, VPs, and structural constraints all working together. The improvement over traditional methods is remarkable! #HybridOptimization #Milestone"

**Demo Output**: Before/after optimization visualizations
**Performance Target**: 2x improvement in challenging scenes

---

### **WEEK 4: Incremental Mapping System**

#### **Day 22: Track Management System**

**Objective**: Implement hybrid track management
**Deliverables**:

- Point/line/VP track data structures
- Track creation and continuation
- Track merging and splitting
- Memory management

**LinkedIn Post**: "📊 Advanced track management system built! Handling thousands of point, line, and VP tracks across image sequences. Data structures matter in computer vision! #TrackManagement #DataStructures"

**Key Features**:

- Efficient track lookup
- Multi-feature track relationships
- Track quality assessment
- Garbage collection

**Success Metric**: Handle 1000+ tracks efficiently

---

#### **Day 23: Incremental Triangulation**

**Objective**: Implement incremental feature triangulation
**Deliverables**:

- On-demand triangulation
- Multi-view triangulation updates
- Triangulation quality assessment
- Failed triangulation handling

**LinkedIn Post**: "⚡ Incremental triangulation working! Adding new 3D features as new images arrive. Building the 3D world one frame at a time! #IncrementalSfM #Triangulation"

**Key Implementation**:

- Triangulation scheduling
- Quality-based triangulation decisions
- Robust triangulation algorithms
- Performance optimization

**Success Metric**: Real-time triangulation of new features

---

#### **Day 24: Keyframe Selection**

**Objective**: Implement intelligent keyframe selection
**Deliverables**:

- Uncertainty-based keyframe scoring
- Coverage-based selection
- Redundancy detection
- Keyframe graph construction

**LinkedIn Post**: "🎯 Smart keyframe selection implemented! Using uncertainty and coverage metrics to choose the most informative frames. Quality over quantity! #KeyframeSelection #SmartSampling"

**Key Features**:

- Multi-criteria keyframe scoring
- Temporal spacing constraints
- Feature coverage analysis
- Graph-based relationships

**Success Metric**: 50% fewer keyframes with same accuracy

---

#### **Day 25: Map Refinement Pipeline**

**Objective**: Integrate local and global refinement
**Deliverables**:

- Adaptive refinement scheduling
- Refinement quality monitoring
- Performance optimization
- Refinement failure handling

**LinkedIn Post**: "🔧 Map refinement pipeline complete! Automatically triggering local and global optimizations based on reconstruction quality. Self-maintaining 3D maps! #MapRefinement #AutoOptimization"

**Key Implementation**:

- Refinement trigger conditions
- Multi-threaded refinement
- Quality improvement tracking
- Rollback mechanisms

**Success Metric**: Stable map quality over long sequences

---

#### **Day 26: Incremental Mapper Integration**

**Objective**: Integrate all mapping components
**Deliverables**:

- Complete incremental mapping system
- Real-time performance optimization
- Memory management
- Error handling and recovery

**LinkedIn Post**: "🚀 Incremental mapping system complete! Real-time 3D reconstruction from image sequences. Points, lines, and VPs building beautiful 3D worlds! #IncrementalMapping #RealTime3D"

**Demo Output**: Live reconstruction from image sequence
**Performance Target**: Process 1 fps image streams

---

#### **Day 27: Mapping System Testing**

**Objective**: Comprehensive testing of mapping system
**Deliverables**:

- Benchmark dataset testing
- Performance profiling
- Memory usage analysis
- Quality metrics collection

**LinkedIn Post**: "📈 Mapping system benchmarked! Testing on ETH3D, Hypersim, and TUM datasets. The numbers are looking great - hybrid features really work! #Benchmarking #Performance"

**Success Metrics**:

- Process 1000+ image sequences
- 20% improvement over point-only
- <2GB memory usage

---

### **WEEK 5: Advanced Features & Integration**

#### **Day 28: Camera Registration Pipeline**

**Objective**: Implement robust camera registration
**Deliverables**:

- Multi-solver registration
- Registration quality scoring
- Failure detection and recovery
- Bootstrap sequence handling

**LinkedIn Post**: "📷 Advanced camera registration complete! Multi-solver RANSAC with hybrid features makes registration incredibly robust. Bringing cameras into the 3D world! #CameraRegistration #PoseEstimation"

**Key Features**:

- Hybrid RANSAC framework
- Pose verification pipeline
- Registration confidence scoring
- Sequential registration

**Success Metric**: 95% registration success rate

---

#### **Day 29: Visualization & Analysis Tools**

**Objective**: Create comprehensive visualization system
**Deliverables**:

- 3D reconstruction viewer
- Feature track visualization
- Uncertainty visualization
- Performance analysis dashboard

**LinkedIn Post**: "👀 Amazing visualization tools built! 3D point clouds, line segments, vanishing points, and uncertainty maps all in one viewer. Seeing is believing! #Visualization #3DReconstruction"

**Key Features**:

- Interactive 3D viewer
- Multi-feature rendering
- Real-time updates
- Export capabilities

**Success Metric**: Clear, informative visualizations

---

#### **Day 30: Python API & High-Level Interface**

**Objective**: Create user-friendly Python API
**Deliverables**:

- Clean Python interface
- Pipeline configuration
- Batch processing tools
- Documentation and examples

**LinkedIn Post**: "🐍 User-friendly Python API complete! High-level interface makes hybrid SfM accessible to researchers and developers. Easy to use, powerful results! #PythonAPI #UserExperience"

**Key Features**:

- Simple pipeline configuration
- Pythonic data structures
- Error handling and logging
- Example notebooks

**Success Metric**: 10-line code for basic reconstruction

---

### **WEEK 6: Testing & Optimization**

#### **Day 31-32: Comprehensive Testing**

**Objective**: Test system on multiple datasets
**Deliverables**:

- ETH3D benchmark results
- Hypersim evaluation
- TUM dataset testing
- Custom dataset validation

**LinkedIn Post**: "📊 Comprehensive evaluation complete! Our hybrid SfM system shows consistent 20-40% improvement over COLMAP on challenging indoor scenes. The research is validated! #Evaluation #Research"

**Success Metrics**:

- Beat COLMAP on indoor scenes
- Match COLMAP on textured outdoor scenes
- Demonstrate robustness improvements

---

#### **Day 33-34: Performance Optimization**

**Objective**: Optimize for production use
**Deliverables**:

- Multi-threading optimization
- Memory usage reduction
- GPU utilization improvement
- Profiling and bottleneck removal

**LinkedIn Post**: "⚡ Performance optimization complete! Multi-threading, GPU acceleration, and memory efficiency make our hybrid SfM system production-ready. Speed meets accuracy! #Optimization #Performance"

**Target Improvements**:

- 2x faster processing
- 50% less memory usage
- Better GPU utilization

---

#### **Day 35: Documentation & Release**

**Objective**: Prepare for open-source release
**Deliverables**:

- Complete documentation
- Installation guides
- Example tutorials
- Performance benchmarks

**LinkedIn Post**: "🎉 Hybrid Structure-from-Motion system complete! 35 days of intensive development, combining classical computer vision with modern deep learning. Open-source release coming soon! #OpenSource #Research #Achievement"

**Final Deliverables**:

- Complete working system
- Comprehensive documentation
- Benchmark results
- Example applications

---

## 📊 Progress Tracking

### Completed Days: [ 2 / 35 ] ✅✅

**Current Phase**: Foundation & Basic Infrastructure  
**Current Day**: Day 3 - Image Data Pipeline  
**Next Milestone**: Basic SfM Pipeline (Day 7)

### Key Metrics to Track:

- [ ] Performance vs COLMAP
- [ ] Memory usage optimization
- [ ] Processing speed (fps)
- [ ] Reconstruction quality
- [ ] Robustness on challenging scenes

### LinkedIn Post Performance:

- Engagement rate: Track likes, comments, shares
- Professional connections made
- Industry interest generated
- Collaboration opportunities

---

## 🔄 Context for Next Chat

**When starting a new chat, share this information:**

1. **Current Day**: 3 - Image Data Pipeline
2. **Completed Tasks**:
   - Day 1: Project structure with CMake, basic types, Python bindings
   - Day 2: 10 camera models with distortion, COLMAP compatibility
3. **Environment**: Ubuntu 22.04, OpenCV 4.13.0-dev, CUDA 12.4, RTX 4070
4. **Project Location**: `~/iSFM`
5. **Virtual Environment**: `hybrid_sfm_env`
6. **Next Objective**: Multi-format image loader with pyramid generation and GPU ops

**Key Files to Mention**:

**Already Created (Days 1-2):**

- `CMakeLists.txt` - Main build configuration
- `src/core/types/camera.h/cpp` - Camera models with distortion
- `src/core/types/image.h/cpp` - Basic image class (needs enhancement)
- `src/core/types/common.h` - Common type definitions
- `src/core/utils/colmap_io.h/cpp` - COLMAP file I/O
- `bindings/types_binding.cpp` - Python bindings
- `tests/cpp/test_camera.cpp` - C++ camera tests
- `tests/python/test_camera.py` - Python camera tests

**To Be Enhanced/Created (Day 3):**

- `src/core/types/image.h/cpp` - Add pyramid generation, GPU operations
- `src/core/utils/image_io.h/cpp` - Multi-format image loading
- `src/core/cuda/image_ops.cu` - CUDA kernels for image operations
- `tests/cpp/test_image.cpp` - Image processing tests
- `tests/python/test_image.py` - Python image tests

**Key Achievements So Far**:

- Solid C++/Python foundation
- Complete camera system with distortion
- Ready for image processing pipeline

---

## 📝 Day-by-Day Progress Notes:

### Day 1: Project Structure & Build System ✅

**Completed**: Successfully set up complete project structure on Ubuntu 22.04

- Created CMakeLists.txt with all dependencies (OpenCV, Eigen, Ceres, pybind11)
- Implemented basic C++ data types (Camera, CameraPose, Image)
- Created working Python bindings with pybind11
- Fixed Eigen alignment issues in CameraPose implementation
- **Technical Decision**: Using standard Ubuntu package management instead of vcpkg
- **Success**: Clean compilation and successful Python module import

### Day 2: Camera Models & Calibration ✅

**Completed**: Implemented comprehensive camera system

- 10 different camera models (COLMAP compatible)
- Full distortion support (radial, OpenCV, fisheye)
- COLMAP I/O compatibility for seamless dataset integration
- Robust projection/unprojection with sub-pixel accuracy
- Complete unit tests (C++ with GTest, Python)
- **Key Achievement**: 19.94 pixel distortion effect demonstrated
- **Success**: All tests passing, ready for image pipeline

- Day 2: [Add notes here]
