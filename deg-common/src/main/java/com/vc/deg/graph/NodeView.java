package com.vc.deg.graph;

import com.vc.deg.FeatureVector;

public interface NodeView {

	public int getLabel();
	public FeatureVector getFeature();
	public WeightedEdges getNeighbors();
}
