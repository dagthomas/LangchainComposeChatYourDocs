const baseFastapiUrl = import.meta.env.VITE_BASE_FASTAPI_URL
const bearerToken = import.meta.env.VITE_BEARER_TOKEN
console.log(bearerToken)
export const load = async ({ parent }) => {
	const { collections } = await parent();
	return { collections };
};

export const actions = {
	upload: async ({ request }) => {
		const formData = await request.formData();
		const response = await fetch(`${baseFastapiUrl}/upload`, {
			method: 'POST', // or 'PUT'
			headers: {
				Authorization: `Bearer ${bearerToken}`
			},
			body: formData
		});

		if (!response.ok) {
			return { success: false };
		}
		return { success: true, result: await response.json() };
	},
	webpage: async ({ request }) => {
		const data = await request.formData();
		const link = data.get('link');

		const response = await fetch(`${baseFastapiUrl}/webpage`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				Authorization: `Bearer ${bearerToken}`
			},
			body: JSON.stringify({
				url: link
			})
		});

		if (!response.ok) {
			return { success: false };
		}
		return { success: true, result: await response.json() };
	},
	webpages: async ({ request }) => {
		const formdata = await request.formData();
		let collection_name = formdata.get('collection_name');
		const data = Object.fromEntries(formdata);

		let cleanedData = Object.keys(data)
			.map(function (key, value) {
				if (key !== 'collection_name') {
					if (
						/^(http(s):\/\/.)[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)$/g.test(
							data[key].toString()
						) === true
					) {
						return data[key];
					}
				}
			})
			.filter((item) => !!item);
		if (collection_name === '') {
			collection_name = `collection-${Math.floor(Math.random() * 100000)}`;
		}
		console.log(collection_name);
		const response = await fetch(`${baseFastapiUrl}/webpages`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				Authorization: `Bearer ${bearerToken}`
			},
			body: JSON.stringify({
				urls: cleanedData,
				collection_name: collection_name
			})
		});

		if (!response.ok) {
			return { success: false };
		}
		return { success: true, result: await response.json() };
	}
};
