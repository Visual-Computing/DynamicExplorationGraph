#pragma once

// deglib Search Module Umbrella Header
//
// Includes all components for querying Dynamic Exploration Graphs:
// - result_list.h: SearchResults, ResultList, ObjectDistance
// - kmeans.h:       KMeansEntrySelector (query-routing entry selection)
// - recall_table.h: RecallTable (parameter recall calibration)
// - reranker.h:     Reranker (high-precision candidate reranker)
// - searcher.h:     Searcher (stateless end-to-end query coordinator)

#include "deglib/search/kmeans.h"
#include "deglib/search/recall_table.h"
#include "deglib/search/reranker.h"
#include "deglib/search/result_list.h"
#include "deglib/search/searcher.h"
