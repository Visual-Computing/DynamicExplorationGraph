package com.vc.deg;

import java.nio.file.Path;

public class GraphFactoryImpl implements GraphFactory {

	@Override
	public DynamicExplorationGraph newGraph(FeatureSpace space, int edgesPerNode) {
		return new DynamicExplorationGraphImpl(space, edgesPerNode);
	}

	@Override
	public DynamicExplorationGraph loadGraph(Path file) {
		// TODO Auto-generated method stub
		return null;
	}

}

