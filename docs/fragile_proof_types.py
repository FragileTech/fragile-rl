"""
Local Sphinx extension to extend ``sphinx_proof`` with directive types used in this book.

This project uses custom proof environments (e.g. ``{prf:metatheorem}``) that are not
provided by ``sphinx_proof`` out of the box. Register them as enumerable nodes so they:
- render as proof-style admonitions,
- participate in numbering (``numfig``), and
- can be referenced with ``{prf:ref}``.
"""

from __future__ import annotations

from docutils import nodes
from sphinx.application import Sphinx


class metatheorem_node(nodes.Admonition, nodes.Element):
    pass


class principle_node(nodes.Admonition, nodes.Element):
    pass


def _register_proof_type(
    app: Sphinx,
    proof_nodes,
    directive_cls: type,
    name: str,
    node_cls: type[nodes.Element],
) -> None:
    proof_nodes.NODE_TYPES.setdefault(name, node_cls)
    app.add_enumerable_node(
        node_cls,
        name,
        None,
        html=(proof_nodes.visit_enumerable_node, proof_nodes.depart_enumerable_node),
        latex=(proof_nodes.visit_enumerable_node, proof_nodes.depart_enumerable_node),
    )
    app.add_directive_to_domain("prf", name, directive_cls)


def setup(app: Sphinx):
    try:
        from sphinx_proof import nodes as proof_nodes
        from sphinx_proof.directive import ElementDirective
        from sphinx_proof.proof_type import PROOF_TYPES
    except Exception:
        return {"version": "builtin", "parallel_read_safe": True, "parallel_write_safe": True}

    class MetatheoremDirective(ElementDirective):
        name = "metatheorem"

    class PrincipleDirective(ElementDirective):
        name = "principle"

    PROOF_TYPES.setdefault("metatheorem", MetatheoremDirective)
    PROOF_TYPES.setdefault("principle", PrincipleDirective)

    _register_proof_type(
        app,
        proof_nodes,
        MetatheoremDirective,
        "metatheorem",
        metatheorem_node,
    )
    _register_proof_type(
        app,
        proof_nodes,
        PrincipleDirective,
        "principle",
        principle_node,
    )

    return {"version": "builtin", "parallel_read_safe": True, "parallel_write_safe": True}
