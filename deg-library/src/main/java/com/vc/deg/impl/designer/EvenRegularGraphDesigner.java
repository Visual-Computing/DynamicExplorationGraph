package com.vc.deg.impl.designer;

import com.vc.deg.GraphDesigner;
import com.vc.deg.FeatureVector;
import com.vc.deg.impl.graph.EvenRegularWeightedUndirectedGraph;

/**
 * Verbessert und erweitert den Graphen
 * 
 * @author Neiko
 *
 */
public class EvenRegularGraphDesigner implements GraphDesigner {
	
	protected EvenRegularWeightedUndirectedGraph graph;

	public EvenRegularGraphDesigner(EvenRegularWeightedUndirectedGraph graph) {
		this.graph = graph;
	}

	/**
	 * Add this new item to the graph
	 * 
	 * @param id
	 */
	public void add(int id, FeatureVector data) {


	}
	
	public void extendGraph(int id, boolean randomPosition) {
		if(randomPosition)
			randomGraphExtension(id);
		
		
		// TODO anns entry point should be always the same node
		
	}
	
	protected void randomGraphExtension(int id) {
		// TODO use worst edges instead of random edge
	}

}
