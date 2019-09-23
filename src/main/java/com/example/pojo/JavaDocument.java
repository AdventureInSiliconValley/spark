package com.example.pojo;

public class JavaDocument {
    private final long id;
    private final String text;

    public JavaDocument(long id, String text) {
        this.id = id;
        this.text = text;
    }

    public long getId() {
        return id;
    }

    public String getText() {
        return text;
    }
}
