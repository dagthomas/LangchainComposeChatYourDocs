export const load = async ({ params, parent }) => {
	const { collections } = await parent();

	return { collection: params.collection, collections: collections };
};
