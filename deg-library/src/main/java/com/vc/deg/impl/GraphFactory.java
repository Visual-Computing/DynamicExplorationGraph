package com.vc.deg.impl;

import java.io.IOException;
import java.nio.file.Path;

import com.vc.deg.FeatureSpace;
import com.vc.deg.HierarchicalDynamicExplorationGraph;

public class GraphFactory implements com.vc.deg.GraphFactory {

	@Override
	public DynamicExplorationGraph newGraph(FeatureSpace space, int edgesPerNode) {
		return new DynamicExplorationGraph(space, edgesPerNode);
	}

	@Override
	public DynamicExplorationGraph loadGraph(Path file) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public com.vc.deg.DynamicExplorationGraph loadGraph(Path file, String featureType) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public HierarchicalDynamicExplorationGraph newHierchicalGraph(FeatureSpace space, int edgesPerNode,
			int topRankSize) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public HierarchicalDynamicExplorationGraph loadHierchicalGraph(Path file) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}
}