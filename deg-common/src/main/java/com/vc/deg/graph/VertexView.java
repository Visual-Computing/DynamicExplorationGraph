package com.vc.deg.graph;

import com.vc.deg.FeatureVector;

public interface VertexView {

	public int getId();
	public FeatureVector getFeature();
	public WeightedEdges getNeighbors();
}
