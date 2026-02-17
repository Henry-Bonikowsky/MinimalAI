package com.minimalai.util;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * Static helpers for float[] <-> NDArray conversion.
 */
public final class TensorUtil {

    private TensorUtil() {}

    /**
     * Create an NDArray from a flat float[] with the given shape.
     */
    public static NDArray toNDArray(NDManager mgr, float[] data, long... shape) {
        return mgr.create(data, new Shape(shape));
    }

    /**
     * Flatten an NDArray to a float[].
     */
    public static float[] toFloatArray(NDArray array) {
        return array.toFloatArray();
    }

    /**
     * Create a zeroed NDArray with the given shape.
     */
    public static NDArray zeros(NDManager mgr, long... shape) {
        return mgr.zeros(new Shape(shape), DataType.FLOAT32);
    }
}
