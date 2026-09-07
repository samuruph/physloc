"""Packaging a generated release for distribution.

Deliberately does NOT re-export `export`: the function and its module share a
name, so `from .export import export` here makes `physloc.release.export`
resolve to the function and shadows the module for every importer.
"""
