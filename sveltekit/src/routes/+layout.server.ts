
const baseFastapiUrl = import.meta.env.VITE_BASE_FASTAPI_URL
const bearerToken = import.meta.env.VITE_BEARER_TOKEN
export const load = async ({ }) => {
	try {
		const response = await fetch(`${baseFastapiUrl}/collections`, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json',
				Authorization: `Bearer ${bearerToken}`
			}
		});
		if (response.status === 404) {
			throw new Error('Page not found');
		} else if (response.status === 500) {
			throw new Error('Server error');
		} else if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}

		return { collections: await response.json() };
	} catch (error) {
		return {};
	}
};
