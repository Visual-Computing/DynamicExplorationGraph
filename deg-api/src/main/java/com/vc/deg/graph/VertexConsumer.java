package com.vc.deg.graph;

import com.vc.deg.FeatureVector;

/**
 * Iterates vertices in a collection
 * 
 * @author Nico Hezel
 */
public interface VertexConsumer {

    /**
     * Performs this operation on the given vertices.
     * 
	 * @param label
	 * @param feature
	 */
    void accept(int label, FeatureVector feature);
}
