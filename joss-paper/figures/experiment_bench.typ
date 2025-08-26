#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#page(width: auto, height: auto, margin: 2pt)[
    #diagram(
        axes: (ltr, btt),
        spacing: 15pt,
        cell-size: (8mm, 10mm),
        node-stroke: 0.8pt,
        node-corner-radius: 2pt,
        edge-stroke: 0.6pt,
        edge-corner-radius: 5pt,
        label-size: 0.8em,
        label-wrapper: edge => box(
            emph[#edge.label],
            inset: .2em,
            radius: .2em,
            fill: edge.label-fill,
        ),
        mark-scale: 120%,
        {
            node((0, 0), [Control computer], name: <computer>)
            edge(
                "->",
                label: [Configures],
                label-side: left,
                label-pos: 50%,
                label-anchor: "south"
            )
            node((2, 0), [Target device], name: <dut>)
            edge(
                <computer>, "d", <measure>, "->",
                shift: (10pt, 4pt),
                label: [Configures],
                label-side: right,
                label-pos: 50%,
                label-anchor: "north-west",
            )
            node((1, 1), [Measuring instrument], name: <measure>)
            edge(
                <measure>, "r", <dut>, "->",
                label: [Measures],
                label-side: left,
                label-pos: 50%,
                label-anchor: "south",
            )
            edge(
                <measure>, "l", <computer>, "->",
                shift: (4pt, 10pt),
                label: [Sends measurements back],
                label-side: right,
                label-pos: 50%,
                label-anchor: "south",
            )
            edge(
                <computer>, "u", <stimulate>, "->",
                shift: (-10pt, 0pt),
                label: [Configures],
                label-side: left,
                label-pos: 50%,
                label-anchor: "south-west",)
            node((1, -1), [Stimulating instrument], name: <stimulate>)
            edge(
                <stimulate>, "r", <dut>, "->",
                label: [Interacts with],
                label-side: right,
                label-pos: 50%,
                label-anchor: "north"
            )
        }
    )
]
