package com.vc.deg.graph;

import com.vc.deg.MemoryView;

public interface NodeView {

	public int getLabel();
	public MemoryView getFeature();
	public WeightedEdges getNeighbors();
}
