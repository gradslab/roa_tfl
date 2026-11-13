"""Renders unicycle_sinusoid.{mp4,gif,jpg}: unicycles switched on across the sinusoid slice. Run: python unicycle_sinusoid.py"""
from unicycle_anim import render

if __name__ == "__main__":
    render("sin6", "unicycle_sinusoid", "Unicycle, sinusoid path", counts=dict(A=30, B=10, C=9),
           rmins=dict(A=0.45, B=0.3, C=0.45), seed=3, width=8.4, dart=0.1, legend_cols=4)
