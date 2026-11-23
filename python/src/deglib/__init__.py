from . import analysis
from . import builder
from . import cpu
from . import distances
from . import graph
from . import optimization
from . import search
from . import transformation
from . import tuning

from .analysis import analyze_graph, GraphStats, calc_graph_quality
from .builder import (
    build,
    GraphBuilder,
    OptimizationTarget,
    BuilderStatus,
    ProgressCallback,
)
from .cpu import (
    InstructionSet,
    has_avx2,
    has_avx2_vnni,
    has_avx512,
    has_avx512_vnni,
    has_avx512_fp16,
    has_fp16,
)
from .distances import Metric, floats_to_fp16, fp16_to_floats
from .graph import (
    DynamicExplorationGraph,
    create_empty,
    create_dynamic_empty,
    create_random_graph,
    load_graph,
    load_readonly_graph,
    load_dynamic_graph,
    load_mutable_graph,
)
from .optimization import presort, prune_non_mrng_edges, prune_worst_edges
from .search import (
    Searcher,
    SearchResults,
    RecallTable,
    ParetoLine,
    create_searcher,
    Reranker,
    rerank,
    Filter,
)
from .transformation import (
    FeatureTransformation,
    make_quantizer_pipeline,
    make_mips_transform,
    make_l2_normalize_transform,
    make_fast_hadamard_rotation,
    make_evp_quantizer,
    make_scalar_quantizer_int8,
    make_scalar_quantizer_uint8,
    make_scalar_quantizer_uint7,
    make_scalar_quantizer_uint4,
    make_scalar_quantizer_int4,
    make_scalar_quantizer_fp16,
    apply_on_database,
    apply_on_query,
)
from .tuning import (
    DatasetProfile,
    TargetPrediction,
    QuantizationPrediction,
    measure_dataset_profile,
    predict_optimization_target,
    predict_quantization,
)

__version__ = "0.3.0"

__all__ = [
    "__version__",
    # Submodules
    "analysis",
    "builder",
    "cpu",
    "distances",
    "graph",
    "optimization",
    "search",
    "transformation",
    "tuning",
    # Builder & top-level build
    "build",
    "GraphBuilder",
    "OptimizationTarget",
    "BuilderStatus",
    "ProgressCallback",
    # Distances & CPU
    "Metric",
    "InstructionSet",
    "floats_to_fp16",
    "fp16_to_floats",
    "has_avx2",
    "has_avx2_vnni",
    "has_avx512",
    "has_avx512_vnni",
    "has_avx512_fp16",
    "has_fp16",
    # Graph & I/O
    "DynamicExplorationGraph",
    "create_empty",
    "create_dynamic_empty",
    "create_random_graph",
    "load_graph",
    "load_readonly_graph",
    "load_dynamic_graph",
    "load_mutable_graph",
    # Search & Rerank
    "Searcher",
    "SearchResults",
    "RecallTable",
    "ParetoLine",
    "create_searcher",
    "Reranker",
    "rerank",
    "Filter",
    # Optimization
    "presort",
    "prune_non_mrng_edges",
    "prune_worst_edges",
    # Analysis
    "analyze_graph",
    "GraphStats",
    "calc_graph_quality",
    # Transformations & Quantizers
    "FeatureTransformation",
    "make_quantizer_pipeline",
    "make_mips_transform",
    "make_l2_normalize_transform",
    "make_fast_hadamard_rotation",
    "make_evp_quantizer",
    "make_scalar_quantizer_int8",
    "make_scalar_quantizer_uint8",
    "make_scalar_quantizer_uint7",
    "make_scalar_quantizer_uint4",
    "make_scalar_quantizer_int4",
    "make_scalar_quantizer_fp16",
    "apply_on_database",
    "apply_on_query",
    # Tuning
    "DatasetProfile",
    "TargetPrediction",
    "QuantizationPrediction",
    "measure_dataset_profile",
    "predict_optimization_target",
    "predict_quantization",
]
