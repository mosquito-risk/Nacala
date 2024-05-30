const EXAMPLES = [
'images/result-image.png',
'images/result-image.png',
'images/result-image.png',
'images/result-image.png',
'images/result-image.png',
]

const Examples = () => {
    return (
        <section class="hero is-small has-carousel is-white">
            <div class="hero-body">
                <div class="container">
                <figure class="model-image">
                    <img src="images/result-image.png" alt="table results" style={{width: "70%"}} />
                    <figcaption style={{ width: "fitContent", height: "fitContent" }}>
                    Figure 4: Predictions from of different models. The predictions are polygonised and styled by class.
                        </figcaption>
                </figure>
                    {/* <h2 class="title is-3 has-text-centered">Examples</h2>
                    <div id="results-carousel" className="carousel results-carousel">
                        {EXAMPLES.map((img, i) => (
                            <div class="item" key={i}>
                                <img src={img} alt={`img${i}`} />
                            </div>
                        ))}
                    </div>
                    <div class="hero-head"></div>
			        <div class="hero-body"></div>
			        <div class="hero-foot"></div> */}
                </div>
            </div>
        </section>
    )
}