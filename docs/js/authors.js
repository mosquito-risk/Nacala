const AUTHORS = [
    {
        name: "Venkanna Babu Guthula",
        affiliation: 1,
        link: "https://scholar.google.com/citations?user=LY1bSd8AAAAJ&hl=en&oi=ao",
    },
    {
        name: "Stefan Oehmcke",
        affiliation: 1,
        link: "https://scholar.google.de/citations?hl=en&user=k9EWJmcAAAAJ&view_op=list_works",
    },
    {
        name: "Remígio Chilaule",
        affiliation: 2,
        link: "https://scholar.google.com/citations?user=p58Vlu4AAAAJ&hl=en&oi=ao",
    },
    {
        name: "Hui Zhang",
        affiliation: 1,
        link: "https://scholar.google.com/citations?user=nOSslqEAAAAJ&hl=en",
    },
    {
        name: "Nico Lang",
        affiliation: 1,
        link: "https://langnico.github.io/",
    },
    {
        name: "Ankit Kariryaa",
        affiliation: 1,
        link: "https://scholar.google.de/citations?hl=en&user=lwSTZGgAAAAJ&view_op=list_works&sortby=pubdate",
    },
    {
        name: "Johan Mottelson",
        affiliation: 2,
        link: "https://scholar.google.de/citations?user=uyvzvC0AAAAJ&hl=en&oi=ao",
    },
    {
        name: "Christian Igel",
        affiliation: 1,
        link: "https://christian-igel.github.io/",
    },
];


const Authors = () => {
    return (
        <div class="is-size-5 publication-authors">
            <span class="author-block">
                {AUTHORS.map((author, i) => (
                    <span key={i}>
                        <a href={author.link} target="_blank" style={{ color: '#3273dc' }}>
                            {author.name}
                        </a>
                        <sup>{author.affiliation}</sup>
                        {i < AUTHORS.length - 1 ? "," : ""}
                    </span>
                ))}
            </span>
            <div class="is-size-5 publication-authors">
                <span class="author-block">
                    <sup>1</sup>
                    <a href="https://di.ku.dk/english/" style={{ color: '#3273dc' }}>University of Copenhagen</a>,
                </span>
                <span class="author-block">
                    <sup>2</sup>
                    <a href="https://royaldanishacademy.com/en" style={{ color: '#3273dc' }}>Royal Danish Academy</a>
                </span>
            </div>
        </div>
    )
}