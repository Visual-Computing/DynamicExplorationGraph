package com.vc.deg.designer;

import com.vc.deg.data.DataComparator;
import com.vc.deg.designer.GraphDesigner;
import com.vc.deg.graph.EvenRegularWeightedUndirectedGraph;

/**
 * Verbessert und erweitert den Graphen
 * 
 * @author Neiko
 *
 */
public class EvenRegularGraphDesigner implements GraphDesigner {
	
	protected EvenRegularWeightedUndirectedGraph graph;
	protected DataComparator comparator;

	public EvenRegularGraphDesigner(EvenRegularWeightedUndirectedGraph graph, DataComparator comparator) {
		this.graph = graph;
		this.comparator = comparator;
	}

	/**
	 * Add this new item to the graph
	 * 
	 * @param id
	 */
	public void add(int id) {


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
