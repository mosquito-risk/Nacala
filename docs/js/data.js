const Data = () => {
    return (
        <section class="section hero is-white">
            <div class="container is-max-desktop">
                <div class="columns is-centered has-text-centered">
                    <div class="column is-four-fifths">
                        <h2 class="title is-3">Nacala-Roof-Material Dataset</h2>
                        <div class="content has-text-justified">
                            <ul>
                                <li>The Nacala-Roof-Material data set provides very high resolution(≈ <b>4.4 cm</b>) drone imagery from three informal
                                    settlements in Nacala in Mozambique, with manually verified, highly accurate annotations of building geometry and roof material.
                                </li>
                                <li>
            Aerial imagery was collected using a DJI Phantom 4 Pro drone and processed using AgiSoft Metashape software. All data were recorded between October and December 2021. All images were made available in <a href='https://openaerialmap.org/' style={{ color: "#3273dc" }} target="_blank">OpenAerialMap</a> with slightly decreased resolution. The drone flight reports for the four areas are in the files <a href='papers/Ontupaia1.pdf' style={{ color: "#3273dc" }} target="_blank">Ontupaia1.pdf</a>, <a href='papers/Ontupaia2.pdf' style={{ color: "#3273dc" }} target="_blank">Ontupaia2.pdf</a>, <a href='papers/Mocone.pdf' style={{ color: "#3273dc" }} target="_blank">Mocone.pdf</a>, and <a href='papers/Ribaue.pdf' style={{ color: "#3273dc" }} target="_blank">Ribaue.pdf</a>.
                                </li>

                                <li>There were <b>17954 buildings</b> in the study area. We distinguished five major types of roof materials in
                                    Nacala, namely <b>metal sheet, thatch, asbestos, concrete</b>, and <b>no-roof</b>, and their counts were 9776, 6428,
                                    566, 174, and 1010, respectively.
                                </li>
                                <li>Each building in this dataset is annotated with a polygon representing the building area and corresponding roof material class. Building polygons were exported from OpenStreetMap and then manually verfied and corrected for geometry and attributes.
                                </li>
                                <li>
                                    For creating the training, validation, and first test data set, <b>stratified sampling</b> was applied to the datat from the first two informal settlements to account for the class imbalance and achieve a similar class distribution in each data split. 
                                </li>
                                <li>
                                    There is <b>no data leaking</b> between sets. If the building area falls into two grid cells and those two cells belong to two different sets (e.g.,
training and test set), we choose to have data pixels in the set where the centroid of the building is
placed and mask the building in the other set.
                                </li>
                                <li> We created <b>an extra geographically separated test set</b> to test the generalization more rigorously.
                                </li>
                                <li>
                                    The data can be downloaded from [<a
                                        href="https://sid.erda.dk/sharelink/aHw1Pey5BC"
                                        style={{ color: "#3273dc" }} target="_blank">here</a>].
                                </li>

                            </ul>

                        </div>

                    </div>

                </div>
                <h2 class="title is-5 has-text-centered" style={{ "paddingTop": "10px" }}>Dataset Overview
                </h2>
                <div>
                    <figure class="model-image">
                        <img src="images/data5CI.png" alt="dataset overview" style={{ width: "90%" }} />
                        <figcaption style={{ width: "fitContent", height: "fitContent" }}>
                            Figure 1: (a) Visualisation of the training, validation and test sets with reference to longitude and latitude;<br />
                            (b) Drone imagery with labels; (c) Instance count of each class for all subsets.
                        </figcaption>
                    </figure>
                </div>
                {/* <div class="fixed-grid has-12-cols is-gap-5">
                    <div class='grid'>
                        <div class='cell is-col-span-4'>
                            <img src="images/train_test_split.png" alt="data split" />
                        </div>
                        <div class='cell is-col-span-8'>
                            <img src="images/data.png" alt="dataset overview" style={{ width: "97%" }} />
                        </div>
                    </div>
                </div> */}
            </div>
        </section>
    )
}
