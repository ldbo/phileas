#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#let cetz = fletcher.cetz
#let draw = cetz.draw

#let multi_layers_rect = (node, extrude, ..parameters) => {
    let step = 2pt
    let steps = 3
    let r = node.corner-radius
    let (w, h) = node.size.map(i => i/2 + extrude)
    draw.group({
        draw.translate(x: step * steps, y: -step * steps)
        for i in range(0, steps) {
            draw.translate(x: -step, y: step)
            draw.fill(white)
            draw.rect(
                (-w, -h), (+w, +h),
                radius: if r != none { r + extrude },
            )
        }
    })
}

#let driver_and_instrument = (y, name) => {
    node((1, y), [Driver])
    edge("->", shift: 4pt)
    edge("<-", shift: -4pt)
    node((2, y), [Instrument])
    let name = label(name)
    node(enclose: ((1, y), (2, y)), stroke: (dash: "dashed"), name: name)
}

#page(width: auto, height: auto, margin: 2pt)[
    #diagram(
        axes: (ltr, btt),
        spacing: 15pt,
        cell-size: (8mm, 10mm),
        node-stroke: 0.8pt,
        node-corner-radius: 0pt,
        edge-stroke: 0.6pt,
        edge-corner-radius: 5pt,
        label-size: 0.8em,
        mark-scale: 100%,
        {
            node(
                (-1.5, 0), [Configurations], name: <configurations>,
                shape: multi_layers_rect,
            )

            node(
                (-0.5, 1.5), [_Apply configurations_], stroke: none
            )
            node(
                (3.5, 1.5), [_Gather measurements_], stroke: none
            )

            driver_and_instrument(1, "instrument0")
            edge(
                <configurations.east>,
                (rel: (0.3, 0)),
                ((), "|-", <instrument0.west>),
                <instrument0.west>,
                "->",
                shift: (-2pt, 0),
            )
            edge(
                <instrument0.east>,
                (<instrument0.east>, "-|", (rel: (-0.3, 0), to: <measurements.west>)),
                (rel: (-0.3, 0), to: <measurements.west>),
                <measurements.west>,
                "->",
                shift: (0pt, -7pt),
            )


            driver_and_instrument(0, "instrument1")
            edge(
                <configurations.east>,
                <instrument1.west>,
                "->",
                shift: (0pt, -2pt),
            )
            edge(
                <instrument1.east>,
                <measurements.west>,
                "->",
                shift: (-5pt, -3pt),
            )


            driver_and_instrument(-1, "instrument2")
            edge(
                <configurations.east>,
                (rel: (0.3, 0)),
                ((), "|-", <instrument2.west>),
                <instrument2.west>,
                "->",
                shift: (2pt, 0pt),
            )

            node(
                (4.5, 0), [Measurements], name: <measurements>,
                shape: multi_layers_rect
            )
            edge("->")
            node((6, 0), [Permanent storage])
        },
    )
]
