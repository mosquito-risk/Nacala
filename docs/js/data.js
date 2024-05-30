const Data = () => {
    return (
        <section class="section hero is-white">
            <div class="container is-max-desktop">
                <div class="columns is-centered has-text-centered">
                    <div class="column is-four-fifths">
                        <h2 class="title is-3">Nacala-Roof-Material Dataset</h2>
                        <div class="content has-text-justified">
                            <ul>
                                <li>Nacala-Roof-Material covers very high resolution(≈ <b>4.4 cm</b>) drone imagery from three informal
                                    settlements of Nacala in Mozambique, with manually verified, highly accurate annotations of building geometry and roof material.
                                </li>
                                <li>
                                Aerial imagery was collected using a DJI Phantom 4 Pro drone and processed using AgiSoft Metashape software. All data was recorded between October and December 2021. We made all images available in <a href='https://openaerialmap.org/' style={{ color: "#3273dc" }} target="_blank">OpenAerialMap</a> with slightly decreased resolution.
                                </li>

                                <li>There are <b>17954 buildings</b> in the study area. We distinguished five major types of roof materials in
                                    Nacala, namely <b>metal sheet, thatch, asbestos, concrete</b>, and <b>no-roof</b>, and their counts are 9776, 6428,
                                    566, 174, and 1010, respectively.
                                </li>
                                <li>Each building in this dataset is annotated with a polygon representing the building area and corresponding roof material class. Building polygons are exported from OpenStreetMap and manually verfied for geometry and attributes.
                                </li>
                                <li>
                                    <b>Stratified sampling</b> was applied to the first two informal settlements to account for the class imbalance and achieve a similar class distribution in each data split. 
                                    We prioritized the distribution of minority classes (i.e., concrete and asbestos)
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
                                        href="https://github.com/mosquito-risk/Nacala"
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
                        <img src="images/data4.png" alt="dataset overview" style={{ width: "90%" }} />
                        <figcaption style={{ width: "fitContent", height: "fitContent" }}>
                            Figure 1: (a) Visualisation of the train, validation and test sets with reference to longitude and latitude;<br />
                            (b) Drone imagery with labels (c) Instance count of each class from all sets
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