"""Renders unicycle_circle.{mp4,gif,jpg}: unicycles switched on across the circle slice. Run: python unicycle_circle.py"""
from unicycle_anim import render

if __name__ == "__main__":
    render("circ3", "unicycle_circle", "Unicycle, circle path", counts=dict(A=26, B=12, C=9, D=3),
           rmins=dict(A=0.4, B=0.22, C=0.3, D=0.12), seed=3)
