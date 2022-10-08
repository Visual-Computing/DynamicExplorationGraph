package com.vc.deg.impl.designer;

import com.vc.deg.GraphDesigner;

import java.util.Random;
import java.util.function.IntPredicate;

import com.vc.deg.FeatureVector;
import com.vc.deg.impl.graph.WeightedUndirectedRegularGraph;

/**
 * Verbessert und erweitert den Graphen
 * 
 * @author Neiko
 *
 */
public class EvenRegularGraphDesigner implements GraphDesigner {
	
	protected WeightedUndirectedRegularGraph graph;

	public EvenRegularGraphDesigner(WeightedUndirectedRegularGraph graph) {
		this.graph = graph;
	}

	/**
	 * Add this new item to the graph
	 * 
	 * @param id
	 */
	public void add(int id, FeatureVector data) {
		// TODO Auto-generated method stub
	}
	
	public void extendGraph(int id, boolean randomPosition) {
		if(randomPosition)
			randomGraphExtension(id);
		
		
		// TODO anns entry point should be always the same node
		
	}
	
	protected void randomGraphExtension(int id) {
		// TODO use worst edges instead of random edge
	}

	@Override
	public void remove(int label) {
		// TODO Auto-generated method stub
	}
	
	@Override
	public void removeIf(IntPredicate filter) {
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
	public float calcAvgEdgeWeight() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setRandom(Random rnd) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public boolean checkGraphValidation(int expectedVertices, int expectedNeighbors) {
		// TODO Auto-generated method stub
		return false;
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

}
