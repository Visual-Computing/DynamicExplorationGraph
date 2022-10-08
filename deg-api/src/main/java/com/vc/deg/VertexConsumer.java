package com.vc.deg;

/**
 * Iterates vertices in a collection
 * 
 * @author Nico Hezel
 */
public interface VertexConsumer {

    /**
     * Performs this operation on the given vertices.
     * 
	 * @param id
	 * @param feature
	 */
    void accept(int id, FeatureVector feature);
}
