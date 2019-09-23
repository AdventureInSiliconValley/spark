package com.example.pojo;

public class JavaLabeledDocument {
    private final long id;
    private final String text;
    private final double label;

    public JavaLabeledDocument(long id, String text, double label) {
        this.id = id;
        this.text = text;
        this.label = label;
    }

    public long getId() {
        return id;
    }

    public String getText() {
        return text;
    }

    public double getLabel() {
        return label;
    }
}