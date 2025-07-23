
def quote_tattoo():
    st.header("Quote Tattoo")
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True, width=200)

    if CROP_AVAILABLE:
        uploaded_image = st.file_uploader("Upload Tattoo Reference", type=["jpg", "png", "jpeg"])

        if uploaded_image:
            image = Image.open(uploaded_image).convert("RGB")
            st.write("Crop the tattoo image to focus on the design:")

            cropped_img = st_cropper(
                image,
                realtime_update=True,
                box_color='#FF0004',
                aspect_ratio=None
            )

            st.image(cropped_img, caption="Cropped Tattoo Image", use_container_width=True)
            image = cropped_img
        else:
            st.warning("Please upload an image to start.")
            return
    else:
        img = st.file_uploader("Upload Tattoo Image", type=["jpg","jpeg","png"])
        if img:
            image = Image.open(img).convert("RGB")
            st.image(image, use_container_width=True)
        else:
            st.warning("Please upload an image to start.")
            return

    artist_filter = st.selectbox("Filter by Artist", ["All"] + settings["artists"])
    compare_count = st.slider("Number of similar tattoos to compare", 1, 5, 3)
    currency = st.selectbox("Currency", ["ZAR", "USD", "EUR"])
    rates = get_live_rates("ZAR")

    temp_path = os.path.join(IMAGE_DIR, "uploaded_image.png")
    image.save(temp_path)

    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        q_feats = model.get_image_features(**inputs)

    df = load_data()
    df = df[~df["artist"].isin(settings.get("archived_artists", []))]
    if artist_filter != "All":
        df = df[df["artist"] == artist_filter]

    feats, rows = [], []
    for _, r in df.iterrows():
        path = os.path.join(IMAGE_DIR, r["filename"])
        if os.path.exists(path):
            ref = Image.open(path).convert("RGB")
            in_ref = processor(images=ref, return_tensors="pt", padding=True)
            with torch.no_grad():
                feats.append(model.get_image_features(**in_ref).squeeze(0))
            rows.append(r)

    if feats:
        sims = torch.nn.functional.cosine_similarity(q_feats, torch.stack(feats))
        dfm = pd.DataFrame(rows)
        dfm["sim"] = sims.tolist()
        dfm.sort_values("sim", ascending=False, inplace=True)
        top_matches = dfm.head(compare_count)
        min_price = top_matches["price"].min()
        max_price = top_matches["price"].max()

        st.subheader("Top Matches")
        st.write(f"**Price Range:** R{min_price} - R{max_price}")
        converted_range = (convert_price(min_price, currency, rates), convert_price(max_price, currency, rates))
        if currency != "ZAR":
            st.write(f"**Converted Price Range:** {currency} {converted_range[0]:.2f} - {currency} {converted_range[1]:.2f}")

        for _, match in top_matches.iterrows():
            st.markdown(f"### {match['artist']} - {match['style']}")
            st.write(f"Price: R{match['price']} ({currency} {convert_price(match['price'], currency, rates):.2f})")
            st.write(f"Time: {match['time']} hrs")
            st.image(os.path.join(IMAGE_DIR, match["filename"]), use_container_width=True)
            st.markdown("---")

        if st.button("📥 Download Quote Report (PDF)") :
            pdf_path = generate_pdf_report(temp_path, top_matches, (min_price, max_price), currency, converted_range)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="tattoo_quote_report.pdf")

        new_log = pd.DataFrame({"date":[pd.to_datetime(date.today())], "artist":[top_matches.iloc[0]['artist']]})
        save_logs(pd.concat([logs, new_log], ignore_index=True))
    else:
        st.warning("No samples available.")
