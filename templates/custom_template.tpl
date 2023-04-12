{% extends 'full.tpl' %}

{% block any_cell %}
{% if 'hide_output' in cell.metadata.tags %}
    <div style="display:none;">
        {{ super() }}
    </div>
{% else %}
    {{ super() }}
{% endif %}
{% endblock any_cell %}

{% block header %}
{{ super() }}
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
{% endblock header %}

{% block body %}
<h2>Table of Contents</h2>
<div id="toc"></div>
<hr>
{{ super() }}
{% endblock body %}

{% block footer %}
{{ super() }}
<script>
$(document).ready(function () {
    let toc = $("#toc");
    let toc_list = $("<ol></ol>");
    $("h2, h3").each(function () {
        let header = $(this);
        let anchor = $("<a></a>");
        let id = header.text().toLowerCase().replace(/\s+/g, '-');
        header.attr("id", id);
        anchor.attr("href", "#" + id);
        anchor.text(header.text());
        let toc_item = $("<li></li>").append(anchor);
        if (header.prop("tagName") === "H3") {
            toc_item.css("margin-left", "20px");
        }
        toc_list.append(toc_item);
    });
    toc.append(toc_list);
});
</script>
{% endblock footer %}
