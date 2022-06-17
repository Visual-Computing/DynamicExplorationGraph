package com.vc.deg.ref;

import java.nio.file.Path;

import com.vc.deg.FeatureSpace;

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

}

