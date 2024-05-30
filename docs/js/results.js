const Results = () => {
    return (
        <div>
            <section class="section hero is-white">
                <div class="container is-max-desktop">
                    <div class="columns is-centered has-text-centered">
                        <div class="column is-four-fifths">
                            <h2 class="title is-3">Experimental Results</h2>
                            <div class="content has-text-justified">
                                Our main results can be summarized as:
                                <ol type="1">
                                    <li>The dataset defines a multi-task problem. No method conducted in this work performs better than the others across all metrics. The
                                        U-Nets and YOLO did well on their home grounds: YOLO gave good object detection results (e.g.,
                                        the best AP50 scores), while the U-Nets performed well for semantic segmentation as measured by
                                        IoU. DINOv2 combined with a simple decoder was also competitive </li>

                                    <li>The proposed deep ordinal watershed (DOW) approach performs generally better in a manner that solves semantic segmentation as well as object detection and classification simultaneously with a high accuracy.
                                    </li>
                                    <li>The DOW idea is applicable beyond the Nacala-Roof-Material data, on which
                                        it improved both thestandard U-Net architectures as well as a system based on DINOv2 features for
                                        segmentation.</li>
                                </ol>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
            <section class="section hero is-white">
                <figure class="model-image">
                    <img src="images/result-table.png" alt="table results" style={{ width: "60%" }} />
                </figure>
            </section>
        </div>
    );
}