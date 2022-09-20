package com.vc.deg.ref.hierarchy;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import com.vc.deg.DynamicExplorationGraph;
import com.vc.deg.FeatureVector;
import com.vc.deg.GraphDesigner;
import com.vc.deg.HierarchicalDynamicExplorationGraph;
import com.vc.deg.SearchResult;

public class HierarchicalGraphDesigner implements GraphDesigner {
	
	protected final List<DynamicExplorationGraph> layers;
	
	public HierarchicalGraphDesigner(List<DynamicExplorationGraph> layers) {
		this.layers = layers;
	}

	@Override
	public void setRandom(Random rnd) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void add(int label, FeatureVector data) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void remove(int label) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void build(ChangeListener listener) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void stop() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setExtendK(int k) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setExtendEps(float eps) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setImproveK(int k) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setImproveEps(float eps) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setMaxPathLength(int maxPathLength) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public float calcAvgEdgeWeight() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public boolean checkGraphValidation(int expectedVertices, int expectedNeighbors) {
		// TODO Auto-generated method stub
		return false;
	}


}
